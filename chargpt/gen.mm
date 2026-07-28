#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/message.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
// argv: model units adapterKey adapterURL emb.bin vocab.txt seed ntokens temp
int main(int argc,char**argv){@autoreleasepool{ setvbuf(stdout,NULL,_IONBF,0);
  NSURL* mu=[NSURL fileURLWithPath:@(argv[1])]; int units=atoi(argv[2]);
  MLModelConfiguration* cfg=[MLModelConfiguration new]; cfg.computeUnits=(MLComputeUnits)units;
  if(strcmp(argv[3],"none")!=0){ NSDictionary* d=@{@(argv[3]):[NSURL fileURLWithPath:@(argv[4])]};
    ((void(*)(id,SEL,id))objc_msgSend)(cfg,NSSelectorFromString(@"setE5rtMutableMILWeightURLs:"),d);}
  NSError* e=nil; MLModel* m=[MLModel modelWithContentsOfURL:mu configuration:cfg error:&e];
  if(!m){printf("LOAD FAIL: %s\n", e?e.localizedDescription.UTF8String:"?"); return 1;}
  // vocab
  FILE* vf=fopen(argv[6],"rb"); fseek(vf,0,SEEK_END); long V=ftell(vf); fseek(vf,0,SEEK_SET);
  std::vector<unsigned char> vocab(V); fread(vocab.data(),1,V,vf); fclose(vf);
  int stoi[256]; for(int i=0;i<256;i++) stoi[i]=-1; for(int i=0;i<V;i++) stoi[vocab[i]]=i;
  // shapes
  MLFeatureDescription* fd=m.modelDescription.inputDescriptionsByName.allValues.firstObject;
  NSArray* sh=fd.multiArrayConstraint.shape; int D=[sh[1] intValue], BLK=[sh[3] intValue];
  NSString* inname=m.modelDescription.inputDescriptionsByName.allKeys.firstObject;
  NSString* outname=m.modelDescription.outputDescriptionsByName.allKeys.firstObject;
  // emb (V*D fp32)
  FILE* ef=fopen(argv[5],"rb"); std::vector<float> emb((size_t)V*D); fread(emb.data(),4,(size_t)V*D,ef); fclose(ef);
  // seed
  std::vector<int> ids; for(const char* p=argv[7]; *p; p++){ int t=stoi[(unsigned char)*p]; if(t>=0) ids.push_back(t);}
  if(ids.empty()) ids.push_back(stoi[(int)'\n']>=0?stoi[(int)'\n']:0);
  int ntok=atoi(argv[8]); double temp=atof(argv[9]); srand48(1234);
  for(size_t i=0;i<ids.size();i++) putchar(vocab[ids[i]]);
  for(int step=0; step<ntok; step++){
    int c=(int)ids.size(); if(c>BLK) c=BLK; int start=(int)ids.size()-c;
    MLMultiArray* x=[[MLMultiArray alloc] initWithShape:@[@1,@(D),@1,@(BLK)] dataType:MLMultiArrayDataTypeFloat32 error:&e];
    float* xp=(float*)x.dataPointer; long sd=x.strides[1].longValue, sp=x.strides[3].longValue;
    memset(xp,0,(size_t)D*BLK*sizeof(float));
    for(int i=0;i<c;i++){ const float* er=&emb[(size_t)ids[start+i]*D]; for(int dd=0;dd<D;dd++) xp[dd*sd+i*sp]=er[dd]; }
    id<MLFeatureProvider> o=[m predictionFromFeatures:[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{inname:x} error:&e] error:&e];
    if(!o){printf("\nPREDICT FAIL: %s\n", e?e.localizedDescription.UTF8String:"?"); return 2;}
    MLMultiArray* y=[[o featureValueForName:outname] multiArrayValue]; int pos=c-1;
    double mx=-1e30; std::vector<double> lg(V);
    for(int v=0;v<V;v++){ lg[v]=[y[@[@0,@(v),@0,@(pos)]] doubleValue]; if(lg[v]>mx) mx=lg[v]; }
    double sum=0; for(int v=0;v<V;v++){ lg[v]=exp((lg[v]-mx)/temp); sum+=lg[v]; }
    double r=drand48()*sum, acc=0; int nx=V-1; for(int v=0;v<V;v++){ acc+=lg[v]; if(acc>=r){nx=v;break;} }
    ids.push_back(nx); putchar(vocab[nx]);
  }
  printf("\n"); return 0; }}
