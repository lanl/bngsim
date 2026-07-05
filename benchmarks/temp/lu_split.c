// Per-call GETRF and GETRS timing (built-in vs Accelerate) for a given N.
// usage: lu_split N
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
extern void dgetrf_(const int*,const int*,double*,const int*,int*,int*);
extern void dgetrs_(const char*,const int*,const int*,const double*,const int*,
                    const int*,double*,const int*,int*);
static long sun_GETRF(double**a,long m,long n,long*p){long i,j,k,l;double*cj,*ck,t,mu,akj;
    for(k=0;k<n;k++){ck=a[k];l=k;for(i=k+1;i<m;i++)if(fabs(ck[i])>fabs(ck[l]))l=i;p[k]=l;
        if(ck[l]==0.0)return k+1;if(l!=k)for(i=0;i<n;i++){t=a[i][l];a[i][l]=a[i][k];a[i][k]=t;}
        mu=1.0/ck[k];for(i=k+1;i<m;i++)ck[i]*=mu;
        for(j=k+1;j<n;j++){cj=a[j];akj=cj[k];if(akj!=0.0)for(i=k+1;i<m;i++)cj[i]-=akj*ck[i];}}return 0;}
static void sun_GETRS(double**a,long n,long*p,double*b){long i,k,pk;double*ck,t;
    for(k=0;k<n;k++){pk=p[k];if(pk!=k){t=b[k];b[k]=b[pk];b[pk]=t;}}
    for(k=0;k<n-1;k++){ck=a[k];for(i=k+1;i<n;i++)b[i]-=ck[i]*b[k];}
    for(k=n-1;k>0;k--){ck=a[k];b[k]/=ck[k];for(i=0;i<k;i++)b[i]-=ck[i]*b[k];}b[0]/=a[0][0];}
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
int main(int argc,char**argv){
    int n=argc>1?atoi(argv[1]):240; long n2=(long)n*n;
    long reps=(long)(3e9/((double)n*n*n)); if(reps<50)reps=50; if(reps>50000)reps=50000;
    double*master=malloc(n2*sizeof(double)); srand(7);
    for(long i=0;i<n2;i++)master[i]=((double)rand()/RAND_MAX)-0.5;
    for(int d=0;d<n;d++)master[(long)d*n+d]+=n;
    // pre-factor copies for GETRS-only timing
    double*w=malloc(n2*sizeof(double)),**cols=malloc(n*sizeof(double*));
    long*p=malloc(n*sizeof(long)); double*b=malloc(n*sizeof(double));
    // builtin GETRF (incl. matrix rebuild via memcpy, as CVODE refills each setup)
    double t0=now_s();
    for(long r=0;r<reps;r++){memcpy(w,master,n2*sizeof(double));for(int c=0;c<n;c++)cols[c]=w+(long)c*n;sun_GETRF(cols,n,n,p);}
    double bf=(now_s()-t0)/reps*1e6;
    // builtin GETRS only (factor once, solve many)
    memcpy(w,master,n2*sizeof(double));for(int c=0;c<n;c++)cols[c]=w+(long)c*n;sun_GETRF(cols,n,n,p);
    t0=now_s();
    for(long r=0;r<reps*4;r++){for(int i=0;i<n;i++)b[i]=1.0+(i%7);sun_GETRS(cols,n,p,b);}
    double bs=(now_s()-t0)/(reps*4)*1e6;
    // lapack GETRF
    double*wl=malloc(n2*sizeof(double));int*ip=malloc(n*sizeof(int));int N=n,LDA=n,INFO,NRHS=1,LDB=n;
    t0=now_s();
    for(long r=0;r<reps;r++){memcpy(wl,master,n2*sizeof(double));dgetrf_(&N,&N,wl,&LDA,ip,&INFO);}
    double lf=(now_s()-t0)/reps*1e6;
    // lapack GETRS only
    memcpy(wl,master,n2*sizeof(double));dgetrf_(&N,&N,wl,&LDA,ip,&INFO);
    double*bl=malloc(n*sizeof(double));
    t0=now_s();
    for(long r=0;r<reps*4;r++){for(int i=0;i<n;i++)bl[i]=1.0+(i%7);dgetrs_("N",&N,&NRHS,wl,&LDA,ip,bl,&LDB,&INFO);}
    double ls=(now_s()-t0)/(reps*4)*1e6;
    printf("N=%d builtin_getrf=%.3fus builtin_getrs=%.3fus | lapack_getrf=%.3fus lapack_getrs=%.3fus | getrf_speedup=%.2fx\n",
           n,bf,bs,lf,ls,bf/lf);
    return 0;
}
