#include <dfitsio2.h>
#include <wcs.h>
#include <math.h>

typedef struct WorldCoor WCS;
//int DEBUG=1;
int DEBUG=0;

float
stddev_calc(int n, float ave, float *data)
{
  int i; float Sxx=0., sig;

  for(i=0; i<n; i++) Sxx +=(data[i]-ave)*(data[i]-ave);

  if(n<=1) sig =0.;
  else     sig =sqrt(Sxx/(float)(n-1)/(float)n); //0817変更 
  // else     sig =sqrt(Sxx/(float)(n-1));

  return(sig);
}

float
median_calc(int n, float *data)
{
  float temp, ret;
  int i, j;

  // the following two loops sort the array x in ascending order
  for(i=0; i<n-1; i++) {
    for(j=i+1; j<n; j++) {
      if(data[j] < data[i]) {
	temp=data[i]; data[i]=data[j]; data[j]=temp; // swap elements
      }
    }
  }

  // if there is an even number of elements, return mean of the two elements in the middle
  // else return the element in the middle
  if(n%2==0) ret = (data[n/2] + data[n/2-1]) / 2.0;
  else       ret = data[n/2];

  return(ret);
}

void
avestd_calc(int n, float ave, float sig, float thresh, float *data,
	    float *ave_rtn, float *sig_rtn, int *n_rtn)
{
  int i;
  int nc=0; float Sx=0., Sxx=0;

  for(i=0; i<n; i++){
    if(data[i]<ave-thresh*sig || ave+thresh*sig<data[i]) continue;
    nc++;
    Sx += data[i];
    Sxx+= data[i]*data[i];
  }

  (*n_rtn)  = nc;
  if(nc==0){
    (*ave_rtn) = 0;
    (*sig_rtn) = 0;
  }else{
    (*ave_rtn) = Sx/ (float)nc;
    (*sig_rtn) = sqrt(Sxx/(float)nc - (*ave_rtn)*(*ave_rtn));
  }

  return;
}

void
avestd_calc_w_weight(int n, float ave, float sig, float thresh, float *data, float *weight,
		     float *ave_rtn, float *sig_rtn, float *weight_rtn, int *n_rtn)
{
  int i;
  int nc=0; float Sx=0., Sxx=0, Wx=0.;

  for(i=0; i<n; i++){
    if(data[i]<ave-thresh*sig || ave+thresh*sig<data[i]) continue;
    nc++;
    Wx += weight[i];
    Sx += data[i];
    Sxx+= data[i]*data[i];
  }

  (*n_rtn)  = nc;
  if(nc==0){
    (*ave_rtn) = 0;
    (*sig_rtn) = 0;
    (*weight_rtn) = 0;
  }else if(nc==1){ // 0819
    (*ave_rtn) = Sx/ (float)nc;
    (*sig_rtn) = 0;
    (*weight_rtn) = Wx/ (float)nc;
  }else{
    (*ave_rtn) = Sx/ (float)nc;
    (*sig_rtn) = Sxx/(float)nc - (*ave_rtn)*(*ave_rtn);
    if ( (*sig_rtn) < 0.0 ){
      (*sig_rtn) = 0.0;
    }
    (*sig_rtn) = sqrt( (*sig_rtn) / (float)(nc-1) );
    (*weight_rtn) = Wx/ (float)nc;
  }

  return;
}

WCS *
wcs_set(Header *h)
{
  WCS *wcs=NULL;

  int naxis1=0;     // Number of pixels along x-axis
  int naxis2=0;     // Number of pixels along y-axis
  char *ctype1=NULL;   // FITS WCS projection for axis 1
  char *ctype2=NULL;   // FITS WCS projection for axis 2
  double crpix1=0.;  // Reference pixel coordinates
  double crpix2=0.;  // Reference pixel coordinates
  double crval1=0.;  // Coordinate at reference pixel in degrees
  double crval2=0.;  // Coordinate at reference pixel in degrees
  double *cd=NULL;     // Rotation matrix, used if not NULL
  double cdelt1=0.;  // scale in degrees/pixel, if cd is NULL
  double cdelt2=0.;  // scale in degrees/pixel, if cd is NULL
  double crota=0.;   // Rotation angle in degrees, if cd is NULL
  int equinox=0;    // Equinox of coordinates, 1950 and 2000 supported
  double epoch=0.;   // Epoch of coordinates

  int err;
  double eq;
  
  naxis1 = header_int_search_by_key(h,"NAXIS1",&err);
  naxis2 = header_int_search_by_key(h,"NAXIS2",&err);

  ctype1 = header_str_search_by_key(h,"CTYPE1");
  ctype2 = header_str_search_by_key(h,"CTYPE2");

  crpix1 = header_double_search_by_key(h,"CRPIX1",&err);
  crpix2 = header_double_search_by_key(h,"CRPIX2",&err);
  crval1 = header_double_search_by_key(h,"CRVAL1",&err);
  crval2 = header_double_search_by_key(h,"CRVAL2",&err);

  eq =  header_double_search_by_key(h,"EQUINOX",&err);
  equinox = (int) eq;
  epoch = 2000;

  cd = (double *)malloc(sizeof(double)*4);
  cd[0] = header_double_search_by_key(h,"CD1_1",&err);
  cd[1] = header_double_search_by_key(h,"CD1_2",&err);
  cd[2] = header_double_search_by_key(h,"CD2_1",&err);
  cd[3] = header_double_search_by_key(h,"CD2_2",&err);

  wcs = wcskinit(naxis1,naxis2, ctype1,ctype2, crpix1,crpix2,
		 crval1,crval2, cd, cdelt1,cdelt2, crota, equinox,epoch);
  return(wcs);
}

void
read_tanzaku(char *filename, int NX, int NY, int NT,
	     float **arr, int *n, struct WorldCoor *wcsG)
{
  FitsFile *ff=NULL; Header *hf=NULL;
  FitsFile *fw=NULL; Header *hw=NULL;
  int nx,ny,nz; float *df=NULL, *dw=NULL;
  char str[256];

  ff = ffile_new_from_file(filename);
  ffile_1blk_fimg_unpack(ff,&hf,FLOAT_IMG,(void **)&df,&nx,&ny,&nz);

  strcpy(str,filename);
  str[strlen(str)-5]='\0';
  strcat(str, ".weight.fits");
  //printf("%s\n", str);

  fw = ffile_new_from_file(str);
  ffile_1blk_fimg_unpack(fw,&hw,FLOAT_IMG,(void **)&dw,&nx,&ny,&nz);

  int x,y; // 作業領域で
  double lon,lat;
  int xp,yp; // 個々の短冊で
  double xq,yq;
  int i;
  int offscl;
  WCS *wcsL=NULL;
  wcsL = wcs_set(hf);

  // for(yp=0; yp<ny; yp++){
  for(yp=60; yp<ny; yp++){
    for(xp=0; xp<nx; xp++){
      if(dw[nx*yp+xp]==0.0) continue;
      // df[i];
      // memoryにたす
      pix2wcs(wcsL, (double)xp, (double)yp, &lon,&lat);
      wcs2pix(wcsG, lon,lat, &xq,&yq, &offscl);
      x = (int) (xq+0.5);
      y = (int) (yq+0.5);
      if(x<0 || x>=NX-1) continue;
      if(y<0 || y>=NY-1) continue;
      arr[NX*y+x][n[NX*y+x]] = df[nx*yp+xp];
      n[NX*y+x]++;
      //      printf("FF %d %d -> %d %d\n", xp,yp, x,y);
    }
  }

  ffile_remove(ff);
  ffile_remove(fw);

  free(wcsL);

  return;
}

void
read_tanzaku_w_weight(char *filename, int NX, int NY, int NT,
		      float **arr, float **arrw, int *n, struct WorldCoor *wcsG)
{
  FitsFile *ff=NULL; Header *hf=NULL;
  FitsFile *fw=NULL; Header *hw=NULL;
  int nx,ny,nz; float *df=NULL, *dw=NULL;
  char str[256];

  ff = ffile_new_from_file(filename);
  ffile_1blk_fimg_unpack(ff,&hf,FLOAT_IMG,(void **)&df,&nx,&ny,&nz);

  strcpy(str,filename);
  str[strlen(str)-5]='\0';
  strcat(str, ".weight.fits");
  //printf("%s\n", str);

  fw = ffile_new_from_file(str);
  ffile_1blk_fimg_unpack(fw,&hw,FLOAT_IMG,(void **)&dw,&nx,&ny,&nz);

  int x,y; // 作業領域で
  double lon,lat;
  int xp,yp; // 個々の短冊で
  double xq,yq;
  int i;
  int offscl;
  WCS *wcsL=NULL;
  wcsL = wcs_set(hf);

  // for(yp=0; yp<ny; yp++){
  for(yp=60; yp<ny; yp++){
    for(xp=0; xp<nx; xp++){
      if(dw[nx*yp+xp]==0) continue;
      // df[i];
      // memoryにたす
      pix2wcs(wcsL, (double)xp, (double)yp, &lon,&lat);
      wcs2pix(wcsG, lon,lat, &xq,&yq, &offscl);
      x = (int) (xq+0.5);
      y = (int) (yq+0.5);
      if(x<0 || x>=NX-1) continue;
      if(y<0 || y>=NY-1) continue;
      arr[NX*y+x][n[NX*y+x]] = df[nx*yp+xp];
      arrw[NX*y+x][n[NX*y+x]]= dw[nx*yp+xp];
      n[NX*y+x]++;
      if(n[NX*y+x]>NT){ printf("NT exceed\n"); exit(0);}
      //      printf("FF %d %d -> %d %d\n", xp,yp, x,y);
    }
  }

  ffile_remove(ff);
  ffile_remove(fw);

  free(wcsL);

  return;
}

int
num_of_fits_count(char *filename)
{
  FILE *fp=NULL;
  char str[256];
  int n=0;

  fp = fopen(filename,"r");
  if(!fp){fprintf(stderr,"Err\n"); return(-1);}
  while(fgets(str,256,fp)!=NULL){
    n++;
  }
  fclose(fp);

  return(n);
}

int
main(int argc, char *argv[])
{
  FitsFile *ff=NULL; Header *h=NULL;
  int nx,ny,nz, x,y; float *di=NULL;

  FILE *fp=NULL;
  char fname[512], str[512];
  int nt;

  int *n=NULL;
  WCS *wcsG=NULL;

  int i;
  float **arr=NULL,**arrw=NULL;
  float *m=NULL, *a=NULL, *s=NULL, *w=NULL;

  float ave,sig,weight,rsig; int rn;
  int j;

  char prefix[512], fname_out[512];

  float CF;

  if(argc!=5){
    fprintf(stderr, "Usage: %s prefix prev.fits input_list.txt ch\n", argv[0]);
    exit(0);
  }

  // 前のversionのタイルを読む
  ff = ffile_new_from_file(argv[2]);
  if(!ff) exit(0);
  ffile_1blk_fimg_unpack(ff,&h,FLOAT_IMG,(void **)&di,&nx,&ny,&nz);

  // Prefix
  strcpy(prefix,argv[1]);
  //  prefix[strlen(prefix)-7]='\0';
  // for(x=0; x<nx*ny; x++) di[x]=0.; 

  // Channel
  char ch;
  //ch = prefix[strlen(prefix)-1];
  ch = argv[4][0];
  if     (ch=='S') CF=0.32;
  else if(ch=='L') CF=0.51;
  else{fprintf(stderr, "Unknown channel %c\n", ch); exit(0);}
  printf("%c\n", ch);

  // タイルのwcsを wcsG にセット
  wcsG = wcs_set(h);

  // リスト内の短冊の数を数える
  nt = num_of_fits_count(argv[3]);
  if(nt<=0){ fprintf(stderr, "Err: nt=%d", nt); exit(0);}
  //printf("%d FITS\n",nt);
  // nt = (int) nt/10; // 経験則 1/100でOK
  nt = (int) nt/5;

  if(DEBUG) printf("T01 %d\n", clock()/1000);


  //float *buf=NULL, *bufw=NULL;
  //buf = (float *)malloc(sizeof(float)*nx*ny*nt);
  //memset(buf, sizeof(float)*nx*ny*nt, 0);
  // 作業用メモリの確保
  arr = (float **)malloc(sizeof(float *)*nx*ny);
  arrw= (float **)malloc(sizeof(float *)*nx*ny);
  for(i=0; i<nx*ny; i++){
    //for(i=0; i<nx*ny; i++){
    // arr[i]=&(buf[nt*i]);
    arr[i]  = (float *)malloc(sizeof(float) * nt);
    arrw[i] = (float *)malloc(sizeof(float) * nt);
    for(j=0; j<nt; j++){ arr[i][j]=0.; arrw[i][j]=0.;}
    //memset(arr[i], sizeof(float)*nt, 0); // +33 sec
  }

  if(DEBUG) printf("T02 %d\n", clock()/1000);

  //d = (float *)malloc(sizeof(float) * nx*ny*nt);
  // memset(d, sizeof(float)*nx*ny*nt, 0);
  n = (int *)malloc(sizeof(int) * nx*ny);
  memset(n, sizeof(int)*nx*ny, 0);
  m = (float *)malloc(sizeof(float) * nx*ny);
  memset(m, sizeof(float)*nx*ny, 0);
  a = (float *)malloc(sizeof(float) * nx*ny);
  memset(a, sizeof(float)*nx*ny, 0);
  s = (float *)malloc(sizeof(float) * nx*ny);
  memset(s, sizeof(float)*nx*ny, 0);
  w = (float *)malloc(sizeof(float) * nx*ny);
  memset(w, sizeof(float)*nx*ny, 0);

  // printf("%d Bytes\n",nt*nx*ny*4);

  if(DEBUG) printf("T03 %d\n", clock()/1000);  

  int tempn=0;
  // リスト内の短冊を読み込む
  fp = fopen(argv[3],"r");
  if(!fp){fprintf(stderr, "Can't open \"%s\".\n", argv[3]);}
  while(fgets(str,512,fp)!=NULL){
    sscanf(str,"%s",fname);
    // strcat(fname,"_mixw_remzod.r.fits");
    // printf("%s\n", fname);
    tempn++;
    read_tanzaku_w_weight(fname, nx,ny,nt, arr,arrw, n, wcsG);
  }
  fclose(fp);

  if(DEBUG) printf("T04 %d\n", clock()/1000); 

  // Combine
  for(y=0; y<ny; y++){
    for(x=0; x<nx; x++){
      m[nx*y+x] = median_calc(n[nx*y+x], arr[nx*y+x]);
      sig = stddev_calc(n[nx*y+x], m[nx*y+x], arr[nx*y+x]);
      avestd_calc_w_weight(n[nx*y+x], m[nx*y+x], sig, 5., arr[nx*y+x], arrw[nx*y+x], &ave,&rsig,&weight,&rn); // 3-sig -> 5-sig (changed on 0818)
      a[nx*y+x] = ave;
      s[nx*y+x] = rsig;
      w[nx*y+x] = weight;
      n[nx*y+x] = rn;
    }
  }

  if(DEBUG) printf("T05 %d\n", clock()/1000);  

  // 作業用メモリの解放
  for(i=0; i<nx*ny; i++){ free(arr[i]); free(arrw[i]); }
  free(arr); free(arrw);

  // FITSを書き出す


  // Nscan FITS
  for(i=0; i<nx*ny; i++) di[i] = (float) n[i];
  sprintf(fname_out, "%s_%c_nsamp.fits", prefix, ch);
  //printf("%s\n", fname_out);
  ffile_write(ff, fname_out);

  // Median FITS (temporary)
  for(i=0; i<nx*ny; i++) di[i] = m[i];
  //memcpy(di, m, sizeof(float)*nx*ny);
  sprintf(fname_out, "%s_%c_median.fits", prefix, ch);
  ffile_write(ff, fname_out);

  // Sigma FITS
  for(i=0; i<nx*ny; i++){
    if(n[i]==0 || n[i]==1) di[i] = 0;
    else        di[i] = s[i]*CF; //0817外す ../sqrt((float)n[i]); //0818 xCF
  }
  // memcpy(di, s, sizeof(float)*nx*ny);
  sprintf(fname_out, "%s_%c_err.fits", prefix, ch);
  ffile_write(ff, fname_out);

  // ADU Average (sigma-cliped) FITS
  for(i=0; i<nx*ny; i++) di[i] = a[i];
  // memcpy(di, a, sizeof(float)*nx*ny);
  sprintf(fname_out, "%s_%c_adu.fits", prefix, ch);
  ffile_write(ff, fname_out);

  // MJy/sr Average (sigma-cliped) FITS
  for(i=0; i<nx*ny; i++) di[i] = a[i]*CF;
  // memcpy(di, a, sizeof(float)*nx*ny);
  sprintf(fname_out, "%s_%c_intensity.fits", prefix, ch);
  ffile_write(ff, fname_out);

  // Weight FITS
  for(i=0; i<nx*ny; i++) di[i] = w[i];
  // memcpy(di, a, sizeof(float)*nx*ny);
  sprintf(fname_out, "%s_%c_weight.fits", prefix, ch);
  ffile_write(ff, fname_out);

  // FITS用メモリの解放
  ffile_remove(ff);
  free(n); free(m); free(s); free(w);

  if(DEBUG) printf("T06 %d\n", clock()/1000);  

  exit(0);
}

// gcc -O3 -o combine160820 combine160820.c -I/usr/ircscan/include -L/usr/ircscan/lib -ldfitsio -lwcs -lm

