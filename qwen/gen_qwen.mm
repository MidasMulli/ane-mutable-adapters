// Autoregressive generation on the ANE with a swappable adapter.
//
// gen_qwen <model.mlmodelc> <units> <adapterKey> <adapterURL> <emb.f16> <promptIds.txt> <nGen>
//
// Loads the base+delta model once, injects one adapter via e5rtMutableMILWeightURLs, then loops:
// host-side fp16 embedding lookup -> ANE forward over a 64-token window -> argmax -> append.
// Prints the generated token ids (decode with decode.py). Requires the ane.entitlements (see
// harness/build_harness.sh); an unentitled binary fails the mutable plan build with error -14.
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/message.h>
#include <cstdio>
#include <vector>
int main(int argc, char** argv){@autoreleasepool{ setvbuf(stdout, NULL, _IONBF, 0);
  NSURL* mu = [NSURL fileURLWithPath:@(argv[1])]; int units = atoi(argv[2]);
  MLModelConfiguration* cfg = [MLModelConfiguration new]; cfg.computeUnits = (MLComputeUnits)units;
  if(strcmp(argv[3], "none") != 0){ NSDictionary* d = @{@(argv[3]): [NSURL fileURLWithPath:@(argv[4])]};
    ((void(*)(id, SEL, id))objc_msgSend)(cfg, NSSelectorFromString(@"setE5rtMutableMILWeightURLs:"), d);}
  NSError* e = nil; MLModel* m = [MLModel modelWithContentsOfURL:mu configuration:cfg error:&e];
  if(!m){printf("LOAD FAIL: %s\n", e ? e.localizedDescription.UTF8String : "?"); return 1;}
  const int D = 1024, Tn = 64, V = 151936;
  FILE* fe = fopen(argv[5], "rb"); if(!fe){printf("no emb\n"); return 9;}
  uint16_t* emb = (uint16_t*)malloc((size_t)V * D * 2); fread(emb, 2, (size_t)V * D, fe); fclose(fe);
  std::vector<int> seq; FILE* fp = fopen(argv[6], "r"); int t; while(fscanf(fp, "%d", &t) == 1) seq.push_back(t); fclose(fp);
  int nGen = atoi(argv[7]);
  NSArray<NSNumber*>* sx = @[@1, @(D), @1, @(Tn)];
  printf("TOKENS:");
  for(int g = 0; g < nGen; g++){
    int L = (int)seq.size() < Tn ? (int)seq.size() : Tn;
    MLMultiArray* x = [[MLMultiArray alloc] initWithShape:sx dataType:MLMultiArrayDataTypeFloat16 error:&e];
    uint16_t* xp = (uint16_t*)x.dataPointer; for(long i = 0; i < (long)D * Tn; i++) xp[i] = 0;
    for(int j = 0; j < L; j++){ int tk = seq[seq.size() - L + j]; for(int d = 0; d < D; d++) xp[(long)d * Tn + j] = emb[(long)tk * D + d]; }
    id<MLFeatureProvider> o = [m predictionFromFeatures:[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": x} error:&e] error:&e];
    if(!o){printf(" PREDICT FAIL: %s\n", e ? e.localizedDescription.UTF8String : "?"); return 2;}
    MLMultiArray* y = [[o featureValueForName:@"logits"] multiArrayValue];
    int sv = y.strides[1].intValue, ss = y.strides[3].intValue; int pos = L - 1;
    double best = -1e300; int bi = 0; for(int v = 0; v < V; v++){double val = y[(long)v * sv + pos * ss].doubleValue; if(val > best){best = val; bi = v;}}
    seq.push_back(bi); printf(" %d", bi);
  }
  printf("\n"); free(emb); return 0;}}
