// Correctness check: solve Ax=b with both kernels, report residual ||Ax-b||_inf
// against the ORIGINAL matrix. Settles whether the bench "mismatch" flag was a
// real solver error or a harness artifact.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

extern void dgetrf_(const int*,const int*,double*,const int*,int*,int*);
extern void dgetrs_(const char*,const int*,const int*,const double*,const int*,
                    const int*,double*,const int*,int*);

static long sun_GETRF(double **a, long m, long n, long *p){
    long i,j,k,l; double *cj,*ck,t,mu,akj;
    for(k=0;k<n;k++){ck=a[k];l=k;
        for(i=k+1;i<m;i++) if(fabs(ck[i])>fabs(ck[l])) l=i;
        p[k]=l; if(ck[l]==0.0) return k+1;
        if(l!=k) for(i=0;i<n;i++){t=a[i][l];a[i][l]=a[i][k];a[i][k]=t;}
        mu=1.0/ck[k]; for(i=k+1;i<m;i++) ck[i]*=mu;
        for(j=k+1;j<n;j++){cj=a[j];akj=cj[k];
            if(akj!=0.0) for(i=k+1;i<m;i++) cj[i]-=akj*ck[i];}}
    return 0;
}
static void sun_GETRS(double **a, long n, long *p, double *b){
    long i,k,pk; double *ck,t;
    for(k=0;k<n;k++){pk=p[k]; if(pk!=k){t=b[k];b[k]=b[pk];b[pk]=t;}}
    for(k=0;k<n-1;k++){ck=a[k]; for(i=k+1;i<n;i++) b[i]-=ck[i]*b[k];}
    for(k=n-1;k>0;k--){ck=a[k]; b[k]/=ck[k]; for(i=0;i<k;i++) b[i]-=ck[i]*b[k];}
    b[0]/=a[0][0];
}

static double resid(double *A, int n, double *x, double *b){ // col-major A
    double m=0;
    for(int i=0;i<n;i++){ double s=0;
        for(int j=0;j<n;j++) s+=A[(long)j*n+i]*x[j];
        double r=fabs(s-b[i]); if(r>m) m=r; }
    return m;
}

int main(void){
    int sizes[]={50,200,800};
    for(int s=0;s<3;s++){
        int n=sizes[s]; long n2=(long)n*n;
        double *A=malloc(n2*sizeof(double));
        srand(999+s);
        for(long i=0;i<n2;i++) A[i]=((double)rand()/RAND_MAX)-0.5;
        for(int d=0;d<n;d++) A[(long)d*n+d]+=n;
        double *b0=malloc(n*sizeof(double));
        for(int i=0;i<n;i++) b0[i]=1.0+(i%7);

        // builtin
        double *w=malloc(n2*sizeof(double)); memcpy(w,A,n2*sizeof(double));
        double **cols=malloc(n*sizeof(double*)); for(int c=0;c<n;c++) cols[c]=w+(long)c*n;
        long *p=malloc(n*sizeof(long)); double *xb=malloc(n*sizeof(double));
        memcpy(xb,b0,n*sizeof(double));
        sun_GETRF(cols,n,n,p); sun_GETRS(cols,n,p,xb);
        double rb=resid(A,n,xb,b0);

        // lapack
        double *wl=malloc(n2*sizeof(double)); memcpy(wl,A,n2*sizeof(double));
        int *ip=malloc(n*sizeof(int)); double *xl=malloc(n*sizeof(double));
        memcpy(xl,b0,n*sizeof(double));
        int N=n,NRHS=1,LDA=n,LDB=n,INFO;
        dgetrf_(&N,&N,wl,&LDA,ip,&INFO);
        dgetrs_("N",&N,&NRHS,wl,&LDA,ip,xl,&LDB,&INFO);
        double rl=resid(A,n,xl,b0);

        // compare the two solution vectors directly
        double dmax=0; for(int i=0;i<n;i++){double d=fabs(xb[i]-xl[i]); if(d>dmax)dmax=d;}
        printf("N=%4d  builtin_resid=%.3e  lapack_resid=%.3e  max|xb-xl|=%.3e\n",
               n, rb, rl, dmax);
        free(A);free(b0);free(w);free(cols);free(p);free(xb);free(wl);free(ip);free(xl);
    }
    return 0;
}
