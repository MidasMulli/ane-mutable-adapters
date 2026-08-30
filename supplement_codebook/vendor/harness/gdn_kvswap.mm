// §1 INSTRUMENT -- KV/RECURRENT STATE ACROSS A MID-SEQUENCE ADAPTER SWAP.
//
// THE QUESTION (directive 2026-08-07T20-32-18 §1.1): every swap verified to date ran a FRESH
// FORWARD ON CLEAN INPUT. A live demo is mid-conversation by construction -- the prompt is
// prefilled under the OLD adapter and generation happens under the NEW one. Does the state
// carried across that boundary change the output, and what is the honest label for what comes out?
//
// WHY A NEW BINARY RATHER THAN A FLAG ON gdn_swap_server: every command in the proven server
// calls freshState() on entry (GEN at :128, SELFTEST at :116). The existing protocol CANNOT
// express "prefill under B, rebind, continue on the same state" -- there is no path through it
// that reaches a rebind with a warm state. This is a clone with one added command. The proven
// tool is left untouched.
//
// STRUCTURAL FACT THE MEASUREMENT MUST NOT ASSUME (Rule 32 / directive §1.2 "THE DISCRIMINATOR
// MUST BE THE OUTPUT, NOT THE PLUMBING"): conv/rec/kv are host-side MLMultiArrays threaded through
// each predict as inputs, and reloadChunks() replaces only the MLModel objects. So state SURVIVES
// a rebind by construction. That predicts CACHE-KEPT. It is not a result. The result is whether
// the OUTPUT of the swapped arm equals clean-A, and it is measured here, not deduced.
//
// TEACHER-FORCED, NOT GREEDY, AND THAT IS THE POINT. Under greedy AR the three arms diverge and
// then see DIFFERENT inputs, so any downstream difference is confounded by trajectory. Teacher
// forcing pins every arm to the SAME token at every position, so a CE difference at position p is
// attributable to the weights and the carried state alone. Greedy argmax is recorded alongside as
// a second witness but is not the discriminator.
//
// THE STANDING GATE (CLAUDE.md): on adapter-over-frozen-base, visual inspection of generations is
// an INVERTING instrument -- fluency measures proximity to the base while accuracy measures the
// adapter's contribution, and they pull opposite ways. Hence CE per arm, per position. No sample
// travels without its CE.
//
// ARMS <nprompt> <swap_at> <ids...>
//   Runs THREE arms in ONE resident process on ONE identical token sequence:
//     cleanON   -- adapter ON bound before position 0
//     cleanOFF  -- adapter OFF bound before position 0
//     swapON    -- OFF bound before position 0, rebound to ON immediately before position swap_at,
//                  WITHOUT resetting conv/rec/kv
//   swap_at is the position index before which the rebind fires, so it is also its own control:
//     swap_at=0 must reproduce cleanON exactly (rebind with cold state == clean bind)
//     swap_at>=N must reproduce cleanOFF exactly (rebind never fires within the sequence)
//   Scores positions nprompt-1 .. N-2, i.e. the CE of predicting ids[nprompt] .. ids[N-1].
//
// Everything else -- load(), stepChunks(), reloadChunks(), freshState(), the raw-buffer logit scan
// and its dtype/layout guard -- is carried verbatim from tools/gdn_swap_server.mm.
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <set>
#import <objc/message.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <time.h>
#include <math.h>
static double nowms(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static NSString* KEY=@"@model_path/weights/adapter.bin";
// The trained adapter in gdnmut/gc*_on.bin is the ABSENCE GATE, not a register adapter -- its task
// is one word, COVERED or NOT_COVERED, decided at a single position. Carried from
// tools/gdn_swap_server.mm:22. The two strings differ at their FIRST token, so the verdict is
// decidable at one position and the margin is one subtraction. This margin is the sharp
// discriminator for §1: the whole decision lives at the position immediately after the swap.
static const long TOK_COVERED_ID=91758, TOK_NOT_ID=4045;
static MLModel* load(NSString* p,int u,NSString* url){
  MLModelConfiguration* c=[MLModelConfiguration new]; c.computeUnits=(MLComputeUnits)u;
  if(url){ ((void(*)(id,SEL,id))objc_msgSend)(c,NSSelectorFromString(@"setE5rtMutableMILWeightURLs:"),@{KEY:[NSURL fileURLWithPath:url]}); }
  NSError* e=nil; MLModel* m=[MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:p] configuration:c error:&e];
  if(!m){printf("LOADFAIL %s: %s\n",p.UTF8String,e?e.localizedDescription.UTF8String:"?"); fflush(stdout); exit(1);} return m; }
static MLMultiArray* zed(NSArray<NSNumber*>* sh){ NSError* e=nil; MLMultiArray* a=[[MLMultiArray alloc] initWithShape:sh dataType:MLMultiArrayDataTypeFloat16 error:&e];
  uint16_t* p=(uint16_t*)a.dataPointer; long n=1; for(NSNumber* q in sh)n*=q.longValue; for(long i=0;i<n;i++)p[i]=0; return a; }
static float* rf32(NSString* p,long n){ FILE* f=fopen(p.UTF8String,"rb"); if(!f){printf("no %s\n",p.UTF8String);exit(9);} float* b=(float*)malloc(n*4); fread(b,4,n,f); fclose(f); return b; }

// DIMS ARE COMPILE-TIME SELECTABLE, DEFAULTING TO THE PROVEN 4B SO THE ANCHOR PATH IS
// BYTE-FOR-BYTE UNCHANGED. Build with -DGDN_Q35_08B=1 for the MONOLITHIC Qwen3.5-0.8B:
// same `qwen3_5` architecture, 24L (18 linear / 6 full), hidden 1024, and NCH=1 because it
// places as ONE model rather than four chunks (2361 ANE / 0 CPU measured 2026-08-12).
// CONV_D = 2*KEY_DIM+VALUE_DIM; NV = linear_num_value_heads; NKV = attention kv heads.
// CTX IS THE HARNESS'S OWN CEILING, NOT THE MODEL'S (card: 262,144 native). It sizes the
// kv_caches MLMultiArray, the causal_mask, and the RoPE table read length -- all three must
// match the .mlmodelc or stepChunks fails with a SHAPE mismatch, not a wrong answer:
//   "MultiArray shape (6 x 2 x 2 x 512 x 256) does not match ... (6 x 2 x 2 x 4096 x 256)"
// Overridable at COMPILE time so the reference binary's default stays byte-unchanged.
// A build at -DGDN_CTX=N ALSO needs cos/sin caches regenerated at N (rebuild_gdn_peripherals.py
// writes CTX*ROT floats; a 512-sized table under CTX=4096 reads past the end of the file).
#ifndef GDN_CTX
#define GDN_CTX 512
#endif
#ifdef GDN_Q35_08B
static const int H=1024, CTX=GDN_CTX, ROT=64, NCH=1, NS_CH=18, NA_CH=6;
static const int CONV_D=6144, NV=16, NKV=2;
// IO CONTRACT DIFFERS BETWEEN THE TWO BUILDS. The 4B demo chunks were converted with SHORT
// feature names (h/conv/rec/kv -> oh/oc/orr/ok); gdn4b_lora_convert.py emits the LONG ones and
// leaves the tuple outputs unnamed (var_*). Mapped by SHAPE, verified against metadata.json:
//   var_5295 [1,1,1024] = hidden   var_5298 [18,6144,3] = conv
//   var_5301 [18,16,128,128] = rec var_5304 [6,2,2,512,256] = kv
#define IN_H "hidden_states"
#define IN_C "conv_states"
#define IN_R "recurrent_states"
#define IN_K "kv_caches"
#define IN_COS "cos_cur"
#define IN_SIN "sin_cur"
#define IN_M "causal_mask"
// The monolithic 0.8B export returns an UNNAMED tuple, so coremltools numbers the outputs
// var_NNNN and the numbering SHIFTS whenever the graph changes. The CQ252 conv re-emission
// returns var_5967/5970/5973/5976 instead of var_5295/5298/5301/5304, and a stale name makes
// stepChunks insert a nil into the input dict -- an NSInvalidArgumentException, not a wrong
// answer. Overridable at COMPILE time so the reference binary's defaults stay byte-unchanged.
#ifndef OUT_H
#define OUT_H "var_5295"
#endif
#ifndef OUT_C
#define OUT_C "var_5298"
#endif
#ifndef OUT_R
#define OUT_R "var_5301"
#endif
#ifndef OUT_K
#define OUT_K "var_5304"
#endif
#else
static const int H=2560, CTX=GDN_CTX, ROT=64, NCH=4, NS_CH=6, NA_CH=2;
static const int CONV_D=8192, NV=32, NKV=4;
#define IN_H "h"
#define IN_C "conv"
#define IN_R "rec"
#define IN_K "kv"
#define IN_COS "cos"
#define IN_SIN "sin"
#define IN_M "mask"
#define OUT_H "oh"
#define OUT_C "oc"
#define OUT_R "orr"
#define OUT_K "ok"
#endif
static NSString* CDIR; static NSString* ADIR; static int UNITS;
static std::vector<MLModel*> ch; static MLModel* emb; static MLModel* lmh;
// IMAGE SPLICE (2026-08-14). The decoder never sees a token id -- `emb` turns one into a
// [1,1,D] hidden vector and THAT is what the chunks consume. So a vision tower emitting
// D-wide vectors can occupy sequence positions directly: same buffer, different source.
// HF does exactly this (masked_scatter over image positions) with NO rescale -- verified on
// the real forward: image |v| 5.17 vs text |e| 0.65, a 7.93x ratio that is BY DESIGN.
static float* imgV=nullptr; static int imgN=0; static int imgPos0=-1;
static float* cosC; static float* sinC; static bool curOn=false;

// §3 SCALING ARM needs a THIRD weight state, so the rebind takes an arbitrary blob SUFFIX.
// The bool form is kept as a wrapper so SWAP/SELFTEST/GENX/ARMS are byte-for-byte unchanged --
// a generalisation that quietly alters the proven commands would invalidate every number already
// measured through them.
// D2 (16-46-21) §2: reload only the FIRST nch chunks. The size sweep needs the bound-byte
// total to vary with the model, dtype and blob held fixed; binding 1..NCH chunks does that
// with zero blob construction, so §2.3's invalid-blob hazard cannot apply to this arm.
// nch==NCH is byte-for-byte the proven path, so the anchor still runs through this function.
static double reloadChunksN(NSString* sfx,int nch){
  if(nch<1) nch=1; if(nch>NCH) nch=NCH;
  double t0=nowms();
  for(int i=0;i<nch;i++){
    NSString* mp=[NSString stringWithFormat:@"%@/gdnmut_gc%d.mlmodelc",CDIR,i];
    // CQ157/CQ182 THIRD ARM. Suffix "none" binds NO adapter URL at all, so
    // setE5rtMutableMILWeightURLs is never called. A GENUINELY MUTABLE build must THROW
    // (E5RT ... BlobFileMutabilityInfo ... not provided by SetMutableMILWeightPaths); if it
    // loads happily the declaration was dropped and every swap on it is a silent no-op.
    // MLComputePlan is blind to this distinction, and neither OFF construct substitutes for
    // it -- an all-zero blob is still a blob. Any OTHER suffix takes the byte-for-byte
    // proven path, so every number already measured through SWAP/SELFTEST/GENX/ARMS stands.
    NSString* url = [sfx isEqualToString:@"none"] ? nil
                  : [NSString stringWithFormat:@"%@/gc%d_%@.bin",ADIR,i,sfx];
    ch[i]=load(mp,UNITS,url);
  }
  return nowms()-t0;
}
static double reloadChunksSuffix(NSString* sfx){ return reloadChunksN(sfx,NCH); }
static double reloadChunks(bool on){ return reloadChunksSuffix(on?@"on":@"off"); }
static MLMultiArray* stepChunks(int tokid,int pos,std::vector<MLMultiArray*>& conv,std::vector<MLMultiArray*>& rec,std::vector<MLMultiArray*>& kv){
  NSError* e=nil; int pp=pos<CTX?pos:CTX-1;
  MLMultiArray* tid=[[MLMultiArray alloc] initWithShape:@[@1,@1] dataType:MLMultiArrayDataTypeInt32 error:&e]; ((int*)tid.dataPointer)[0]=tokid;
  MLMultiArray* h=[[[emb predictionFromFeatures:[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"token_id":tid} error:&e] error:&e] featureValueForName:@"hidden_states"] multiArrayValue];
  // Overwrite IN PLACE rather than synthesising an array: `emb`'s own output already carries
  // the exact shape and dataType the chunks expect, so nothing about that contract is guessed.
  if(imgV && imgPos0>=0 && pos>=imgPos0 && pos<imgPos0+imgN){
    const long D=(long)h.count; const float* src=imgV+(long)(pos-imgPos0)*D;
    if(h.dataType==MLMultiArrayDataTypeFloat16){ uint16_t* d=(uint16_t*)h.dataPointer;
      for(long q=0;q<D;q++){ __fp16 v=(__fp16)src[q]; memcpy(d+q,&v,2); } }
    else { float* d=(float*)h.dataPointer; for(long q=0;q<D;q++) d[q]=src[q]; }
  }
  MLMultiArray* cosU=[[MLMultiArray alloc] initWithShape:@[@1,@1,@1,@(ROT)] dataType:MLMultiArrayDataTypeFloat16 error:&e];
  MLMultiArray* sinU=[[MLMultiArray alloc] initWithShape:@[@1,@1,@1,@(ROT)] dataType:MLMultiArrayDataTypeFloat16 error:&e];
  uint16_t* cp=(uint16_t*)cosU.dataPointer; uint16_t* sp=(uint16_t*)sinU.dataPointer;
  for(int i=0;i<ROT;i++){ __fp16 cv=(__fp16)cosC[(long)pp*ROT+i]; __fp16 sv=(__fp16)sinC[(long)pp*ROT+i]; memcpy(cp+i,&cv,2); memcpy(sp+i,&sv,2); }
  MLMultiArray* mask=[[MLMultiArray alloc] initWithShape:@[@1,@1,@1,@(CTX)] dataType:MLMultiArrayDataTypeFloat16 error:&e];
  uint16_t* mp=(uint16_t*)mask.dataPointer; int valid=(pos+1<CTX)?(pos+1):CTX;
  for(int i=0;i<CTX;i++){ __fp16 mv=(i>=CTX-valid)?(__fp16)0.0f:(__fp16)-65504.0f; memcpy(mp+i,&mv,2); }
  for(int ci=0; ci<NCH; ci++){
    id<MLFeatureProvider> o=[ch[ci] predictionFromFeatures:[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@IN_H:h,@IN_C:conv[ci],@IN_R:rec[ci],@IN_K:kv[ci],@IN_COS:cosU,@IN_SIN:sinU,@IN_M:mask} error:&e] error:&e];
    if(!o){printf("GC%d FAIL: %s\n",ci,e?e.localizedDescription.UTF8String:"?"); fflush(stdout); exit(3);}
    h=[[o featureValueForName:@OUT_H] multiArrayValue];
    conv[ci]=[[o featureValueForName:@OUT_C] multiArrayValue]; rec[ci]=[[o featureValueForName:@OUT_R] multiArrayValue]; kv[ci]=[[o featureValueForName:@OUT_K] multiArrayValue];
  }
  return h;
}
static void freshState(std::vector<MLMultiArray*>& conv,std::vector<MLMultiArray*>& rec,std::vector<MLMultiArray*>& kv){
  conv.clear();rec.clear();kv.clear();
  for(int i=0;i<NCH;i++){ conv.push_back(zed(@[@(NS_CH),@(CONV_D),@3])); rec.push_back(zed(@[@(NS_CH),@(NV),@128,@128])); kv.push_back(zed(@[@(NA_CH),@2,@(NKV),@(CTX),@256])); }
}
// CE of `target` under the lm_head at this hidden state, plus the argmax. Raw-buffer scan with the
// SAME dtype/layout guard the server uses; a guard miss returns NAN rather than a wrong number,
// because a CE from a second code path is a second instrument.
static double ceAt(MLMultiArray* h,int target,int* argmaxOut,double* pmaxOut,double* marginOut){
  NSError* e=nil;
  MLMultiArray* y=[[[lmh predictionFromFeatures:[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"hidden_states":h} error:&e] error:&e] featureValueForName:@"logits"] multiArrayValue];
  long V=1; for(NSNumber* q in y.shape) V*=q.longValue;
  const BOOL fast=(y.dataType==MLMultiArrayDataTypeFloat16)&&(y.count==V)&&(((NSNumber*)y.strides.lastObject).longValue==1);
  if(!fast){ if(argmaxOut)*argmaxOut=-1; return NAN; }
  const __fp16* p=(const __fp16*)y.dataPointer;
  float mx=-INFINITY; int bi=0;
  for(long v=0;v<V;v++){ float val=(float)p[v]; if(val>mx){mx=val;bi=(int)v;} }
  double s=0.0; for(long v=0;v<V;v++) s+=exp((double)((float)p[v])-(double)mx);
  if(argmaxOut)*argmaxOut=bi;
  if(pmaxOut)*pmaxOut=1.0/s;                       // softmax prob of the argmax
  // the gate's own decision variable: logit(COVERED) - logit(NOT). Free -- the raw buffer is
  // already mapped and already scanned. This is the sharp discriminator, not mean CE.
  if(marginOut)*marginOut=(V>TOK_NOT_ID)?((double)((float)p[TOK_COVERED_ID])-(double)((float)p[TOK_NOT_ID])):NAN;
  if(target<0||target>=V) return NAN;
  return -(((double)((float)p[target]))-(double)mx-log(s));
}

// One arm. startOn = adapter bound before position 0. swapAt < 0 disables the rebind.
// swapTo = adapter bound immediately before position swapAt, state NOT reset.
static void runArm(const char* nm,bool startOn,int swapAt,bool swapTo,
                   const std::vector<int>& ids,int nprompt){
  std::vector<MLMultiArray*> conv,rec,kv;
  double bindms=reloadChunks(startOn); freshState(conv,rec,kv);
  double swapms=-1.0;
  int N=(int)ids.size(); double tot=0.0, mtot=0.0; int cnt=0, agree=0;
  printf("ARM %s start=%s swap_at=%d swap_to=%s bind_ms=%.1f\n",nm,startOn?"on":"off",swapAt,swapTo?"on":"off",bindms);
  for(int g=0; g<N; g++){
    if(swapAt>=0 && g==swapAt){ swapms=reloadChunks(swapTo); printf("SWAPFIRED at_pos=%d ms=%.1f\n",g,swapms); }
    MLMultiArray* h=stepChunks(ids[g],g,conv,rec,kv);
    if(g<nprompt-1) continue;                       // scoring starts at the last prompt position
    if(g+1>=N) break;                               // no target beyond the sequence
    int am=-1; double pmax=0.0, mg=NAN;
    double ce=ceAt(h,ids[g+1],&am,&pmax,&mg);
    printf("CE %d %d %d %.6f %.6f %.6f\n",g,ids[g+1],am,ce,pmax,mg);
    if(!isnan(ce)){ tot+=ce; cnt++; }
    if(!isnan(mg)) mtot+=mg;
    if(am==ids[g+1]) agree++;
  }
  printf("ARMDONE %s mean_ce=%.6f mean_margin=%.6f n=%d top1=%d swap_ms=%.1f\n",
         nm,cnt?tot/cnt:NAN,cnt?mtot/cnt:NAN,cnt,agree,swapms);
  fflush(stdout);
}

// suffix-aware twin of runArm. Same body, same grading, only the bind target generalised.
static void runArmSfx(const char* nm,NSString* startSfx,int swapAt,NSString* swapSfx,
                      const std::vector<int>& ids,int nprompt){
  std::vector<MLMultiArray*> conv,rec,kv;
  double bindms=reloadChunksSuffix(startSfx); freshState(conv,rec,kv);
  double swapms=-1.0;
  int N=(int)ids.size(); double tot=0.0,mtot=0.0; int cnt=0,agree=0;
  printf("ARM %s start=%s swap_at=%d swap_to=%s bind_ms=%.1f\n",
         nm,startSfx.UTF8String,swapAt,swapSfx.UTF8String,bindms);
  for(int g=0; g<N; g++){
    if(swapAt>=0 && g==swapAt){ swapms=reloadChunksSuffix(swapSfx);
      printf("SWAPFIRED at_pos=%d ms=%.1f\n",g,swapms); }
    MLMultiArray* h=stepChunks(ids[g],g,conv,rec,kv);
    if(g<nprompt-1) continue;
    if(g+1>=N) break;
    int am=-1; double pmax=0.0,mg=NAN;
    double ce=ceAt(h,ids[g+1],&am,&pmax,&mg);
    printf("CE %d %d %d %.6f %.6f %.6f\n",g,ids[g+1],am,ce,pmax,mg);
    if(!isnan(ce)){ tot+=ce; cnt++; }
    if(!isnan(mg)) mtot+=mg;
    if(am==ids[g+1]) agree++;
  }
  printf("ARMDONE %s mean_ce=%.6f mean_margin=%.6f n=%d top1=%d swap_ms=%.1f\n",
         nm,cnt?tot/cnt:NAN,cnt?mtot/cnt:NAN,cnt,agree,swapms);
  fflush(stdout);
}

int main(int argc,char**argv){@autoreleasepool{ setvbuf(stdout,NULL,_IONBF,0);
  CDIR=@(argv[1]); ADIR=@(argv[2]); NSString* embp=@(argv[3]); NSString* lmhp=@(argv[4]); NSString* cachedir=@(argv[5]); UNITS=atoi(argv[6]);
  cosC=rf32([cachedir stringByAppendingString:@"/cos_cache.f32"],(long)CTX*ROT);
  sinC=rf32([cachedir stringByAppendingString:@"/sin_cache.f32"],(long)CTX*ROT);
  ch.resize(NCH);
  emb=load(embp,UNITS,nil); lmh=load(lmhp,UNITS,nil);
  // CQ157 THIRD ARM, AND THE STARTUP BIND IS PART OF THE TEST. A `SWAPN 1 none` issued AFTER
  // the process has already bound a real URL is CONTAMINATED: the plan is warm and the mutable
  // path has already been exercised. The canonical diagnostic (gen_qwen argv[3]="none") makes
  // the FIRST load the one with no URL, so START_SFX does the same here.
  const char* ssfx = getenv("START_SFX"); NSString* startSfx = ssfx ? @(ssfx) : @"off";
  double ms0=reloadChunksSuffix(startSfx); curOn=false;
  printf("READY %s %.1f units=%d\n",startSfx.UTF8String,ms0,UNITS); fflush(stdout);
  char line[1<<16];
  while(fgets(line,sizeof(line),stdin)){
    char* nl=strchr(line,'\n'); if(nl)*nl=0;
    if(!strncmp(line,"Q",1)) break;
    // D2 §2 size sweep. SWAPN <nch> <suffix> binds the first <nch> chunks from
    // gc<i>_<suffix>.bin and reports the wall time. Placed BEFORE the SWAP prefix test
    // because "SWAP" is a prefix of "SWAPN" and strncmp(...,4) would swallow it -- the
    // ordering is load-bearing, not cosmetic.
    if(!strncmp(line,"SWAPN",5)){
      char* t=strtok(line+5," "); int nch=t?atoi(t):NCH;
      t=strtok(NULL," "); NSString* sfx=t?@(t):@"on";
      double ms=reloadChunksN(sfx,nch);
      printf("SWAPPEDN %d %s %.1f\n",nch,[sfx UTF8String],ms); fflush(stdout);
      // curOn is now UNKNOWN for a partial or non-on/off suffix; force the next arm to
      // rebind rather than trust a stale flag.
      curOn=false; continue;
    }
    if(!strncmp(line,"SWAP",4)){
      bool on=strstr(line,"on")!=NULL; double ms=reloadChunks(on); curOn=on;
      printf("SWAPPED %s %.1f\n",on?"on":"off",ms); fflush(stdout); continue;
    }
    // POSITIVE CONTROL, run before any arm: one forward OFF then ON on the same input.
    // A swap that binds without changing the output would make every arm below meaningless.
    if(!strncmp(line,"SELFTEST",8)){
      std::vector<int> ids; char* t=strtok(line+8," "); while(t){ids.push_back(atoi(t));t=strtok(NULL," ");}
      NSError* e=nil; std::vector<MLMultiArray*> conv,rec,kv;
      for(int arm=0;arm<2;arm++){ reloadChunks(arm==1); freshState(conv,rec,kv);
        MLMultiArray* h=nil; for(size_t g=0; g<ids.size(); g++) h=stepChunks(ids[g],(int)g,conv,rec,kv);
        MLMultiArray* y=[[[lmh predictionFromFeatures:[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"hidden_states":h} error:&e] error:&e] featureValueForName:@"logits"] multiArrayValue];
        long V=1; for(NSNumber* q in y.shape) V*=q.longValue;
        const __fp16* p=(const __fp16*)y.dataPointer; double s=0; for(long v=0;v<V;v++) s+=fabs((double)((float)p[v]));
        int am=-1; double pm=0, mg=NAN; ceAt(h,0,&am,&pm,&mg);
        printf("SELF %s out.sumabs=%.4f argmax=%d margin=%.6f\n",arm?"on ":"off",s,am,mg); fflush(stdout);
      }
      reloadChunks(curOn); continue;
    }
    // CAPABILITY PROBE. SCORE <startOn> <tokA> <tokB> <ids...>
    //   One forward over the whole prompt, then the lm_head at the last position. Reports the CE
    //   of tokA, the CE of tokB, and logit(A)-logit(B). A FORCED CHOICE between two candidate
    //   continuations that differ at their first token: the sign of the margin IS the model's
    //   answer, graded exactly, with no generation and no parsing. Chance is 0.
    if(!strncmp(line,"SCORE",5)){
      char* t=strtok(line+5," "); int startOn=t?atoi(t):0;
      t=strtok(NULL," "); int ta=t?atoi(t):-1;
      t=strtok(NULL," "); int tb=t?atoi(t):-1;
      std::vector<int> ids; while((t=strtok(NULL," "))) ids.push_back(atoi(t));
      if(ids.empty()){printf("SCOREERR no ids\n");fflush(stdout);continue;}
      if((startOn!=0)!=curOn){ reloadChunks(startOn!=0); curOn=(startOn!=0); }
      std::vector<MLMultiArray*> conv,rec,kv; freshState(conv,rec,kv);
      MLMultiArray* h=nil;
      for(size_t g=0; g<ids.size(); g++) h=stepChunks(ids[g],(int)g,conv,rec,kv);
      int am=-1; double pmax=0.0, dummy=NAN;
      double cea=ceAt(h,ta,&am,&pmax,&dummy);
      double ceb=ceAt(h,tb,NULL,NULL,NULL);
      // margin = log p(A) - log p(B); the normaliser cancels, so it is a pure pairwise preference
      printf("SC ce_a=%.6f ce_b=%.6f margin=%.6f argmax=%d\n",cea,ceb,ceb-cea,am);
      fflush(stdout); continue;
    }
    // §3 DEMO. GENX <startOn> <swapAt> <swapTo> <nprompt> <ngen> <tgt> <ids...>
    //   Binds `startOn` before position 0, optionally rebinds to `swapTo` immediately before
    //   position `swapAt` WITHOUT resetting state, prefills, then generates `ngen` tokens greedily.
    //   At the FIRST generated position it reports CE of `tgt` and the COVERED-minus-NOT margin, so
    //   every emitted sample travels with its own CE rather than as an impression. `swapAt` < 0
    //   disables the rebind, which is how the two clean control arms are produced by this same path.
    if(!strncmp(line,"GENX",4)){
      char* t=strtok(line+4," "); int startOn=t?atoi(t):0;
      t=strtok(NULL," "); int swapAt=t?atoi(t):-1;
      t=strtok(NULL," "); int swapTo=t?atoi(t):1;
      t=strtok(NULL," "); int nprompt=t?atoi(t):1;
      t=strtok(NULL," "); int ngen=t?atoi(t):8;
      t=strtok(NULL," "); int tgt=t?atoi(t):-1;
      std::vector<int> ids; while((t=strtok(NULL," "))) ids.push_back(atoi(t));
      if((int)ids.size()<nprompt){printf("GENXERR too few ids\n");fflush(stdout);continue;}
      std::vector<MLMultiArray*> conv,rec,kv;
      double bindms=reloadChunks(startOn!=0); freshState(conv,rec,kv);
      double swapms=-1.0;
      printf("GENXBEGIN start=%s swap_at=%d swap_to=%s bind_ms=%.1f\n",
             startOn?"on":"off",swapAt,swapTo?"on":"off",bindms); fflush(stdout);
      std::vector<int> seq(ids.begin(),ids.begin()+nprompt); double t0=nowms(); int made=0;
      for(int g=0; g<nprompt+ngen; g++){
        if(swapAt>=0 && g==swapAt){ swapms=reloadChunks(swapTo!=0);
          printf("SWAPFIRED at_pos=%d ms=%.1f\n",g,swapms); fflush(stdout); }
        int tokid=(g<nprompt)?seq[g]:seq.back();
        MLMultiArray* h=stepChunks(tokid,g,conv,rec,kv);
        if(g<nprompt-1) continue;
        int am=-1; double pmax=0.0, mg=NAN;
        double ce=ceAt(h,(g==nprompt-1)?tgt:-1,&am,&pmax,&mg);
        if(g==nprompt-1) printf("DEC ce=%.6f margin=%.6f argmax=%d pmax=%.6f\n",ce,mg,am,pmax);
        seq.push_back(am); made++;
        printf("TOK %d\n",am); fflush(stdout);
      }
      printf("GENXDONE made=%d ms=%.1f swap_ms=%.1f\n",made,nowms()-t0,swapms); fflush(stdout);
      reloadChunks(curOn); continue;
    }
    // GENI <embfile> <imgPos0> <imgN> <startOn> <nprompt> <ngen> <nstop> <stopids...> <ids...>
    //   Same loop as GENX, except positions [imgPos0, imgPos0+imgN) take their hidden vector
    //   from <embfile> (raw float32, imgN x D) instead of from the embedding table. That is
    //   the whole vision wiring: the ids at those positions are the model's own image
    //   placeholder token and are never looked up.
    //   TERMINATION IS A REPORTED STATE (freeze criterion CI2.1 + CI3): the run stops on a
    //   supplied stop id and reports `term=eos`, or exhausts ngen and reports `term=cap`.
    //   A BIGGER FIXED CAP STILL TRUNCATES, LATER -- the two are different observations and
    //   neither may be inferred from the token count.
    if(!strncmp(line,"GENI",4)){
      char* t=strtok(line+4," "); std::string ef=t?t:"";
      t=strtok(NULL," "); int p0=t?atoi(t):-1;
      t=strtok(NULL," "); int nimg=t?atoi(t):0;
      t=strtok(NULL," "); int startOn=t?atoi(t):0;
      t=strtok(NULL," "); int nprompt=t?atoi(t):1;
      t=strtok(NULL," "); int ngen=t?atoi(t):8;
      t=strtok(NULL," "); int nstop=t?atoi(t):0;
      std::set<int> stopset;
      for(int q=0;q<nstop;q++){ t=strtok(NULL," "); if(t) stopset.insert(atoi(t)); }
      std::vector<int> ids; while((t=strtok(NULL," "))) ids.push_back(atoi(t));
      if((int)ids.size()<nprompt){printf("GENIERR too few ids\n");fflush(stdout);continue;}
      FILE* f=fopen(ef.c_str(),"rb");
      if(!f){printf("GENIERR no embfile %s\n",ef.c_str());fflush(stdout);continue;}
      fseek(f,0,SEEK_END); long nf=ftell(f)/4; fseek(f,0,SEEK_SET);
      if(imgV) free(imgV); imgV=(float*)malloc(nf*4);
      if(fread(imgV,4,nf,f)!=(size_t)nf){printf("GENIERR short read\n");fflush(stdout);fclose(f);continue;}
      fclose(f); imgN=nimg; imgPos0=p0;
      printf("GENIVEC floats=%ld imgN=%d pos0=%d dim=%ld\n",nf,imgN,imgPos0,nimg?nf/nimg:0); fflush(stdout);
      std::vector<MLMultiArray*> conv,rec,kv;
      // Bind ONLY on an arm change -- the SCORE handler's proven conditional, applied to GENI.
      // A repeat request on the already-bound arm reuses the resident weights and reports
      // bind_ms=0.0. curOn is sticky for GENI (no trailing restore; every caller passes startOn).
      // ⛔ If the staged blob BYTES change under the SAME suffix (a re-deploy), the caller MUST
      // issue `SWAP <arm>` first -- the skip is keyed on the arm flag, not on the bytes.
      double bindms=0.0; bool giWant=(startOn!=0);
      if(giWant!=curOn){ bindms=reloadChunks(giWant); curOn=giWant; }
      freshState(conv,rec,kv);
      printf("GENIBEGIN start=%s bind_ms=%.1f\n",startOn?"on":"off",bindms); fflush(stdout);
      std::vector<int> seq(ids.begin(),ids.begin()+nprompt); double t0=nowms(); int made=0;
      const char* term="cap";
      for(int g=0; g<nprompt+ngen; g++){
        int tokid=(g<nprompt)?seq[g]:seq.back();
        MLMultiArray* h=stepChunks(tokid,g,conv,rec,kv);
        if(g<nprompt-1) continue;
        int am=-1; double pmax=0.0, mg=NAN;
        ceAt(h,-1,&am,&pmax,&mg);
        seq.push_back(am); made++;
        printf("TOK %d\n",am); fflush(stdout);
        if(stopset.count(am)){ term="eos"; break; }
      }
      printf("GENIDONE made=%d ms=%.1f term=%s\n",made,nowms()-t0,term); fflush(stdout);
      imgPos0=-1;                      // the splice is per-call, never sticky
      continue;                        // no restore rebind: curOn already reflects the bound arm
    }
    // ARMSX <startSuffix> <nprompt> <swapAt> <ids...>
    //   Three arms on ONE sequence: cleanOFF and cleanON (the interval endpoints) plus a swap
    //   that STARTS on <startSuffix> and rebinds to `on` at <swapAt>. The excursion beyond
    //   [cleanOFF, cleanON] is the graded quantity; at startSuffix=on it must be ZERO by
    //   construction, which is the run's own control.
    if(!strncmp(line,"ARMSX",5)){
      char* t=strtok(line+5," "); std::string sfx = t?t:"off";
      t=strtok(NULL," "); int nprompt=t?atoi(t):1;
      t=strtok(NULL," "); int swapAt=t?atoi(t):-1;
      std::vector<int> ids; while((t=strtok(NULL," "))) ids.push_back(atoi(t));
      if((int)ids.size()<nprompt+1){printf("ARMSXERR too few ids\n");fflush(stdout);continue;}
      NSString* S=[NSString stringWithUTF8String:sfx.c_str()];
      printf("ARMSXBEGIN start=%s nprompt=%d swap_at=%d\n",sfx.c_str(),nprompt,swapAt);
      fflush(stdout);
      runArmSfx("cleanOFF",@"off",-1,@"off",ids,nprompt);
      runArmSfx("cleanON" ,@"on" ,-1,@"on" ,ids,nprompt);
      runArmSfx("swapX"   ,S    ,swapAt,@"on",ids,nprompt);
      printf("ARMSXEND\n"); fflush(stdout);
      reloadChunks(curOn); continue;
    }
    if(!strncmp(line,"ARMS",4)){
      char* t=strtok(line+4," "); int nprompt=t?atoi(t):1;
      t=strtok(NULL," "); int swapAt=t?atoi(t):-1;
      std::vector<int> ids; while((t=strtok(NULL," "))) ids.push_back(atoi(t));
      if((int)ids.size()<nprompt+1){printf("ARMSERR too few ids\n");fflush(stdout);continue;}
      printf("ARMSBEGIN nprompt=%d swap_at=%d ntok=%d\n",nprompt,swapAt,(int)ids.size()); fflush(stdout);
      runArm("cleanON" ,true ,-1     ,true ,ids,nprompt);
      runArm("cleanOFF",false,-1     ,false,ids,nprompt);
      runArm("swapON"  ,false,swapAt ,true ,ids,nprompt);
      // instrument self-controls: a rebind with COLD state must equal a clean bind.
      runArm("ctlSwap0",false,0      ,true ,ids,nprompt);
      printf("ARMSEND\n"); fflush(stdout);
      reloadChunks(curOn); continue;
    }
  }
  return 0;}}
