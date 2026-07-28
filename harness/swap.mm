#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/message.h>
#include <cstdio>
int main(int argc,char**argv){@autoreleasepool{ setvbuf(stdout,NULL,_IONBF,0);
  NSURL* mu=[NSURL fileURLWithPath:@(argv[1])]; int units=atoi(argv[2]);
  MLModelConfiguration* cfg=[MLModelConfiguration new]; cfg.computeUnits=(MLComputeUnits)units;
  if(strcmp(argv[3],"none")!=0){ NSDictionary* d=@{@(argv[3]):[NSURL fileURLWithPath:@(argv[4])]};
    ((void(*)(id,SEL,id))objc_msgSend)(cfg,NSSelectorFromString(@"setE5rtMutableMILWeightURLs:"),d);}
  NSError* e=nil; MLModel* m=[MLModel modelWithContentsOfURL:mu configuration:cfg error:&e];
  if(!m){printf("LOAD FAIL: %s\n", e?e.localizedDescription.UTF8String:"?"); return 1;}
  NSString* in=m.modelDescription.inputDescriptionsByName.allKeys.firstObject;
  MLFeatureDescription* fd=m.modelDescription.inputDescriptionsByName.allValues.firstObject;
  MLMultiArray* x=[[MLMultiArray alloc] initWithShape:fd.multiArrayConstraint.shape dataType:MLMultiArrayDataTypeFloat32 error:&e];
  long n=1; for(NSNumber* q in fd.multiArrayConstraint.shape) n*=q.longValue; for(long i=0;i<n;i++) x[i]=@(((i*37)%101)/101.0 - 0.5);
  id<MLFeatureProvider> o=[m predictionFromFeatures:[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{in:x} error:&e] error:&e];
  if(!o){printf("PREDICT FAIL: %s\n", e?e.localizedDescription.UTF8String:"?"); return 2;}
  NSString* on=m.modelDescription.outputDescriptionsByName.allKeys.firstObject;
  MLMultiArray* y=[[o featureValueForName:on] multiArrayValue];
  int Vn=y.shape[1].intValue, Sn=y.shape[3].intValue, sv=y.strides[1].intValue, ss=y.strides[3].intValue, lp=Sn-1;
  double best=-1e300; int bi=-1; for(int v=0;v<Vn;v++){double val=y[v*sv+lp*ss].doubleValue; if(val>best){best=val;bi=v;}}
  double sa=0; for(long i=0;i<y.count;i++) sa+=fabs(y[i].doubleValue);
  printf("pred_token(last pos)=%d  logit=%.4f  sum_abs=%.2f\n",bi,best,sa); return 0;}}
