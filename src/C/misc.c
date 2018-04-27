// Copyright 2010-2018 M. S. Andersen & L. Vandenberghe
//
// This file is part of SMCP.
//
// SMCP is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// SMCP is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with SMCP.  If not, see <http://www.gnu.org/licenses/>.

//#define DEBUG
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Python.h"
#include "cvxopt.h"

#define PY_ERR(E,str) { PyErr_SetString(E, str); return NULL; }
#define PY_ERR_INT(E,str) { PyErr_SetString(E, str); return -1; }
#define PY_ERR_TYPE(str) PY_ERR(PyExc_TypeError, str)

#ifdef _OPENMP
#include "omp.h"
#endif


PyDoc_STRVAR(misc__doc__,
    "Miscellaneous routines.");

static char doc_sdpa_readhead[] =
  "Reads sparse SDPA data file header.\n"
  "\n"
  "n,m,bstruct = sdpa_readhead(f)\n"
  "\n"
  "PURPOSE\n"
  "Read sparse SDPA data file header information\n"
  "without reading data.\n"
  "\n"
  "ARGUMENTS\n"
  "f         Python file object\n"
  "\n"
  "RETURNS\n"
  "n         Integer\n"
  "\n"
  "m         Integer\n"
  "\n"
  "bstruct   CVXOPT integer matrix\n";

static PyObject* sdpa_readhead
(PyObject *self, PyObject *args)
{
  int i,j,t;
  int_t m=0,n=0,nblocks=0;
  matrix *bstruct = NULL;
  PyObject *f;

  char buf[2048];  // buffer
  char *info;

  if (!PyArg_ParseTuple(args,"O",&f)) return NULL;
#if PY_MAJOR_VERSION >= 3
    if (PyUnicode_Check(f)) {
      const char* fname = PyUnicode_AsUTF8AndSize(f,NULL);
#else
    if (PyString_Check(f)) {
      const char* fname = PyString_AsString(f);
#endif
      FILE *fp = fopen(fname,"r");
      if (!fp) {
        return NULL;
      }
      /* Skip comments and read m */
      while (1) {
        info = fgets(buf,1024,fp);
        if (buf[0] != '*' && buf[0] != '"') {
          sscanf(buf,"%d",&i);
          break;
        }
      }
      m = (int_t) i;

      /* read nblocks */
      j = fscanf(fp,"%d",&i);
      nblocks = (int_t) i;

      /* read blockstruct and compute block offsets*/
      bstruct = Matrix_New(nblocks,1,INT);
      if (!bstruct) return PyErr_NoMemory();
      n = 0;
      for (i=0; i<nblocks; i++) {
        j = fscanf(fp,"%*[^0-9+-]%d",&t);
        MAT_BUFI(bstruct)[i] = (int_t) t;
        n += (int_t) labs(MAT_BUFI(bstruct)[i]);
      }
      fclose(fp);
  }

  return Py_BuildValue("iiN",n,m,bstruct);
}


static char doc_sdpa_read[] =
  "Reads sparse SDPA data file (dat-s).\n"
  "\n"
  "A,b,bstruct = sdpa_read(f[,neg=False])\n"
  "\n"
  "PURPOSE\n"
  "Reads problem data from sparse SDPA data file for\n"
  "the semidefinite programs:\n"
  "\n"
  "  (P)  minimize    <A0,X>\n"
  "       subject to  <Ai,X> = bi,   i = 1,...,m\n"
  "                   X >= 0\n"
  "\n"
  "  (D)  maximize    b'*y\n"
  "       subject to  sum_i Ai*yi + S = A0\n"
  "                   S >= 0\n"
  "\n"
  "Here '>=' means that X and S must be positive semidefinite.\n"
  "The matrices A0,A1,...Am are symmetric and of order n.\n"
  "If the optional argument 'neg' is True, the negative of the\n"
  "problem data is returned.\n"
  "\n"
  "ARGUMENTS\n"
  "f         Python file object\n"
  "\n"
  "neg       Python boolean (optional)\n"
  "\n"
  "RETURNS\n"
  "A         CVXOPT sparse matrix of doubles with columns Ai[:]\n"
  "          (Only lower trianglular elements of Ai are stored.)\n"
  "\n"
  "b         CVXOPT matrix\n"
  "\n"
  "bstruct   CVXOPT integer matrix\n";

static PyObject* sdpa_read
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  int i,j,mno,bno,ii,jj,t;
  int_t k,m,n,nblocks,nlines;
  double v;
  long fpos;
  PyObject *f;
  PyObject *neg = Py_False;
  char *info;
  const char* fname;
  int_t* boff;     // block offset
  char buf[2048];  // buffer
  char *kwlist[] = {"f","neg",NULL};

  if (!PyArg_ParseTupleAndKeywords(args,kwrds,"O|O",kwlist,&f,&neg)) return NULL;
  #if PY_MAJOR_VERSION >= 3
  if (PyUnicode_Check(f)) fname = PyUnicode_AsUTF8AndSize(f,NULL);
  #elif PY_MAJOR_VERSION == 2
  if (PyString_Check(f)) fname = PyString_AsString(f);
  #endif
  FILE *fp = fopen(fname,"r");
  if (!fp) {
    return NULL;
  }
  /* Skip comments and read m */
  while (1) {
    info = fgets(buf,1024,fp);
    if (buf[0] != '*' && buf[0] != '"') {
      sscanf(buf,"%d",&i);
      break;
    }
  }
  m = (int_t) i;

  /* read nblocks */
  j = fscanf(fp,"%d",&i);
  nblocks = (int_t) i;

  /* read blockstruct and compute block offsets*/
  matrix *bstruct = Matrix_New(nblocks,1,INT);
  if (!bstruct) return PyErr_NoMemory();
  boff = malloc(sizeof(int_t)*(nblocks+1));
  if(!boff) return PyErr_NoMemory();
  boff[0] = 0;  n = 0;
  for (i=0; i<nblocks; i++) {
    j = fscanf(fp,"%*[^0-9+-]%d",&t);
    MAT_BUFI(bstruct)[i] = (int_t) t;
    n += (int_t) labs(MAT_BUFI(bstruct)[i]);
    boff[i+1] = n;
  }

  /* read vector b */
  matrix *b = Matrix_New(m,1,DOUBLE);
  if (!b) return PyErr_NoMemory();
  for (i=0;i<m;i++) {
    j = fscanf(fp,"%*[^0-9+-]%lf",&MAT_BUFD(b)[i]);
    if (neg == Py_True)
      MAT_BUFD(b)[i] *= -1;
  }

  /* count remaining lines */
  fpos = ftell(fp);
  for (nlines = 0; fgets(buf, 1023, fp) != NULL; nlines++);
  //nlines--;
  fseek(fp,fpos,SEEK_SET);

  /* Create data matrix A */
  spmatrix *A = SpMatrix_New(n*n,m+1,nlines,DOUBLE);
  if (!A) return PyErr_NoMemory();

  // read data matrices
  fseek(fp,fpos,SEEK_SET);
  for (i=0,j=-1,k=0;k<nlines;k++){
    if (fscanf(fp,"%*[^0-9+-]%d",&mno) <=0 ) break;
    if (fscanf(fp,"%*[^0-9+-]%d",&bno) <=0 ) break;
    if (fscanf(fp,"%*[^0-9+-]%d",&ii) <=0 ) break;
    if (fscanf(fp,"%*[^0-9+-]%d",&jj) <=0 ) break;
    if (fscanf(fp,"%*[^0-9+-]%lf",&v) <=0 ) break;

    // check that value is nonzero
    if (v != 0) {
      // add block offset
      ii += boff[bno-1];
      jj += boff[bno-1];

      // insert index and value
      SP_ROW(A)[i] = (int_t)  ((ii-1)*n + (jj-1));
      if (neg == Py_True)
	SP_VALD(A)[i] = -v;
      else
	SP_VALD(A)[i] = v;

      // update col. ptr.
      while (mno > j)
	SP_COL(A)[++j] = i;

      i++;
    }
  }
  // update last element(s) of col. ptr.
  while (m+1 > j)
    SP_COL(A)[++j] = i;

  fclose(fp);

  // free temp. memory
  free(boff);

  return Py_BuildValue("NNN",A,b,bstruct);
}

// UPDATE DOC STRING!
static char doc_sdpa_write[] =
  "Writes problem data to sparse SDPA data file.\n"
  "\n"
  "sdpa_write(f,A,b,bstruct[,neg=False])\n"
  "\n"
  "PURPOSE\n"
  "Converts and writes problem data associated with the\n"
  "pair of semidefinite programming problems\n"
  "\n"
  "  (P)  minimize    <A0,X>\n"
  "       subject to  <Ai,X> = bi,   i = 1,...,m\n"
  "                   X >= 0\n"
  "\n"
  "  (D)  maximize    b'*y\n"
  "       subject to  sum_i Ai*yi + S = A0\n"
  "                   S >= 0\n"
  "\n"
  "to sparse SDPA data file format (dat-s).\n"
  "Here '>=' means that X and S must be positive semidefinite.\n"
  "The matrices A0,A1,...Am are symmetric and of order n.\n"
  "\n"
  "The block structure is specified with a vector 'bstruct'. The\n"
  "i'th element of bstruct determines the size of i'th block.\n"
  "\n"
  "ARGUMENTS\n"
  "f         Python file object\n"
  "\n"
  "A         CVXOPT sparse matrix with columns Ai[:]\n"
  "          (Only lower triangular elements of Ai are used.)\n"
  "\n"
  "b         CVXOPT matrix\n"
  "\n"
  "bstruct   CVXOPT integer matrix\n"
  "\n"
  "neg       Python boolean (optional)\n";

static PyObject* sdpa_write
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  int i,Il,Jl,Bl,Ml;
  int_t n;
  spmatrix *A;
  matrix *b,*bstruct;
  PyObject *f;
  PyObject *neg = Py_False;
  char *kwlist[] = {"f","A","b","bstruct","neg",NULL};
  const char* fname;
  double v;

  if (!PyArg_ParseTupleAndKeywords(args,kwrds, "OOOO|O", kwlist, &f, &A, &b, &bstruct,&neg)) return NULL;
  #if PY_MAJOR_VERSION >= 3
  if (PyUnicode_Check(f)) fname = PyUnicode_AsUTF8AndSize(f,NULL);
  #elif PY_MAJOR_VERSION == 2
  if (PyString_Check(f)) fname = PyString_AsString(f);
  #endif
  FILE *fp = fopen(fname,"r");
  if (!fp) {
    Py_DECREF(f);
    return NULL;
  }

  fprintf(fp,"* sparse SDPA data file (created by SMCP)\n");
  fprintf(fp,"%i = m\n",(int) MAT_NROWS(b));
  fprintf(fp,"%i = nBlocks\n", (int) MAT_NROWS(bstruct));
  // compute n and write blockstruct
  n = 0;
  for (i=0;i<MAT_NROWS(bstruct);i++) {
    fprintf(fp,"%i ", (int) MAT_BUFI(bstruct)[i]);
    n += (int_t) labs(MAT_BUFI(bstruct)[i]);
  }
  fprintf(fp,"\n");

  // write vector b
  if (neg == Py_True) {
    for (i=0;i<MAT_NROWS(b);i++)
      fprintf(fp,"%.12g ",-MAT_BUFD(b)[i]);
  }
  else {
    for (i=0;i<MAT_NROWS(b);i++)
      fprintf(fp,"%.12g ",MAT_BUFD(b)[i]);
  }
  fprintf(fp,"\n");

  // Write data matrices A0,A1,A2,...,Am
  for (Ml=0;Ml<=MAT_NROWS(b);Ml++) {
    for (i=0;i<SP_COL(A)[Ml+1]-SP_COL(A)[Ml];i++){

      Jl = 1 + SP_ROW(A)[SP_COL(A)[Ml]+i] / n;
      Il = 1 + SP_ROW(A)[SP_COL(A)[Ml]+i] % n;

      // Skip if element is in strict upper triangle
      if (Jl > Il)
	PyErr_Warn(PyExc_Warning,"Ignored strictly upper triangular element.");

      Bl = 1;
      while ((Il > labs(MAT_BUFI(bstruct)[Bl-1])) && (Jl > labs(MAT_BUFI(bstruct)[Bl-1]))) {
	Il -= (int_t) labs(MAT_BUFI(bstruct)[Bl-1]);
	Jl -= (int_t) labs(MAT_BUFI(bstruct)[Bl-1]);
	Bl += 1;
      }
      /* Error check */
      if ((Il > labs(MAT_BUFI(bstruct)[Bl-1])) || (Jl > labs(MAT_BUFI(bstruct)[Bl-1])))
	printf("Error: Matrix contains elements outside blocks!\n");

      // print upper triangle entries:
      //   <matno> <blkno> <i> <j> <entry>
      v = SP_VALD(A)[SP_COL(A)[Ml]+i];
      if ( v != 0.0) {
	if (neg == Py_True)
	  fprintf(fp,"%i %i %i %i %.12g\n",
		  (int) Ml,(int) Bl,(int) Jl,(int) Il, -v);
	else
	  fprintf(fp,"%i %i %i %i %.12g\n",
		  (int) Ml,(int) Bl,(int) Jl,(int) Il, v);
      }
    }
  }

  fclose(fp);
  Py_DECREF(f);
  Py_RETURN_NONE;
}

static char doc_ind2sub[] =
  "Converts vector of absolute indices to row and column indices.\n"
  "\n"
  "I,J = ind2sub(siz,IND)\n"
  "\n"
  "PURPOSE\n"
  "Returns the matrices I and J containing the equivalent row and\n"
  "column subscripts corresponding to each linear index in the matrix\n"
  "IND for a matrix of size siz.\n"
  "\n"
  "ARGUMENTS\n"
  "siz       Integer\n"
  "\n"
  "IND       CVXOPT dense integer matrix\n"
  "\n"
  "RETURNS\n"
  "I         CVXOPT dense integer matrix\n"
  "\n"
  "J         CVXOPT dense integer matrix\n";

static PyObject *ind2sub
(PyObject *self, PyObject *args)
{
  matrix *Im;
  int_t i;
  int_t n;

  if (!PyArg_ParseTuple(args, "nO", &n, &Im)) return NULL;

  matrix *Il = Matrix_New(MAT_NROWS(Im),1,INT);
  if (!Il) return PyErr_NoMemory();
  matrix *Jl = Matrix_New(MAT_NROWS(Im),1,INT);
  if (!Il) return PyErr_NoMemory();

  for (i=0;i< MAT_NROWS(Im);i++) {
    MAT_BUFI(Il)[i] = MAT_BUFI(Im)[i] % n;
    MAT_BUFI(Jl)[i] = MAT_BUFI(Im)[i] / n;
  }

  return Py_BuildValue("NN", Il, Jl);
}

static char doc_sub2ind[] =
  "Converts subscripts to linear index vector.\n"
  "\n"
  "Ind = sub2ind(siz,I,J)\n"
  "\n"
  "PURPOSE\n"
  "Computes linear index from subscripts matrices\n"
  "I and J. The size of the matrix is a tuple siz.\n"
  "\n"
  "ARGUMENTS\n"
  "siz       Integer tuple (m,n)\n"
  "\n"
  "I         CVXOPT integer matrix\n"
  "\n"
  "J         CVXOPT integer matrix\n"
  "\n"
  "RETURNS\n"
  "Ind       CVXOPT integer matrix\n";

static PyObject *sub2ind
(PyObject *self, PyObject *args)
{
  matrix *Im,*Jm;
  PyObject *siz;
  int_t i;
  int_t m,n;

  if (!PyArg_ParseTuple(args, "OOO", &siz, &Im, &Jm)) return NULL;
  if (!PyArg_ParseTuple(siz, "nn", &m, &n)) return NULL;

  matrix *Ind = Matrix_New(MAT_NROWS(Im),1,INT);
  if (!Ind) return PyErr_NoMemory();

  for (i=0;i< MAT_NROWS(Im) ;i++) {
    // Add data check:
    MAT_BUFI(Ind)[i] = MAT_BUFI(Im)[i] + m*MAT_BUFI(Jm)[i];
  }
  return Py_BuildValue("N", Ind);
}

static char doc_Av_to_spmatrix[] =
 "Converts column from sparse matrix to a sparse matrix.\n"
 "\n"
 "Aj = spvec2spmatrix(Av,Ip,Jp,j,n[,scale=False])\n"
 "\n"
 "PURPOSE\n"
 "...\n"
 "The 'scale' keyword is used to enable/disable scaling\n"
 "of the diagonal elements in Aj. 'True' will scale the\n"
 "diagonal elements by 0.5 while 'False' is default and\n"
 "disables scaling.\n"
 "\n"
 "ARGUMENTS\n"
 "Av        CVXOPT spmatrix, |V|-by-m\n"
 "\n"
 "Ip        CVXOPT integer matrix\n"
 "\n"
 "Jp        CVXOPT integer matrix\n"
 "\n"
 "j 	    Integer\n"
 "\n"
 "n         Integer\n"
 "\n"
 "RETURNS\n"
 "Aj        CVXOPT spmatrix, n-by-n\n";

static PyObject *Av_to_spmatrix
(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *scale = Py_False;
  spmatrix *Av,*Ip,*Jp;
  int_t i,j,n,nnz,c,ci,p,q;
  char *kwlist[] = {"Av","Ip","Jp","j","n","scale",NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOOnn|O",
				   kwlist, &Av,&Ip,&Jp,&j,&n, &scale))
    return NULL;

  p = SP_COL(Av)[j];
  nnz = SP_COL(Av)[j+1]-p;
  spmatrix *Aj = SpMatrix_New(n,n,nnz,DOUBLE);
  if (!Aj) return PyErr_NoMemory();

  // Generate col-ptr and row index
  SP_COL(Aj)[0] = 0;
  if (scale==Py_False) {
    for (ci=0,i=0;i<nnz;i++) {
      q = SP_ROW(Av)[p+i];
      SP_ROW(Aj)[i] = MAT_BUFI(Ip)[q];
      c = MAT_BUFI(Jp)[q];
      SP_VALD(Aj)[i] = SP_VALD(Av)[p+i];
      while (ci < c)
	SP_COL(Aj)[++ci] = i;
    }
    while (ci < n)
      SP_COL(Aj)[++ci] = nnz;
  }
  else {
    for (ci=0,i=0;i<nnz;i++) {
      q = SP_ROW(Av)[p+i];
      SP_ROW(Aj)[i] = MAT_BUFI(Ip)[q];
      c = MAT_BUFI(Jp)[q];
      SP_VALD(Aj)[i] = SP_VALD(Av)[p+i];
      if (c == SP_ROW(Aj)[i]) SP_VALD(Aj)[i] *= 0.5; // scale diag. element
      while (ci < c)
	SP_COL(Aj)[++ci] = i;
    }
    while (ci < n)
      SP_COL(Aj)[++ci] = nnz;
  }

  return (PyObject *) Aj;
}


static char doc_scal_diag[] =
  "Scales diagonal elements of sparse matrix.\n"
  "\n"
  "scal_diag(V,Id,t)\n"
  "\n"
  "PURPOSE\n"
  "Scales diagonal elements by the factor t.\n"
  "Conceptually the function performs the following:\n"
  "\n"
  "V.V[Id] *= t\n"
  "\n"
  "ARGUMENTS\n"
  "V         CVXOPT sparse matrix\n"
  "\n"
  "Id        CVXOPT integer matrix\n"
  "\n"
  "t         Python float (optional, default is 0.5)\n";

static PyObject *scal_diag
(PyObject *self, PyObject *args)
{
  PyObject *Vp,*Id;
  int_t i,n;
  double t = 0.5;

  if(!PyArg_ParseTuple(args,"OO|d",&Vp,&Id,&t)) return NULL;
  n = MAT_NROWS(Id);

  for (i=0;i<n;i++){
    SP_VALD(Vp)[MAT_BUFI(Id)[i]] *= t;
  }

  Py_RETURN_NONE;
}


static char doc_SCMcolumn[] =
 "Computes column of the Schur complement matrix.\n"
 "\n"
 "SCMcolumn(H,Av,V,Ip,Jp,j)\n"
 "\n"
 "PURPOSE\n"
 "Computes the lower triangular elements of the j'th\n"
 "column of the Schur complement matrix H.\n"
 "The i'th column of the sparse matrix A is\n"
 "tril(vec(Ai)). V is a matrix with the nonzero\n"
 "columns of Aj (first Kj columns) and the correspon-\n"
 "ing Kj columns of the identity matrix.\n"
 "\n"
 "ARGUMENTS\n"
 "H         CVXOPT matrix, m-by-m\n"
 "\n"
 "Av        CVXOPT sparse matrix, |V|-by-m\n"
 "\n"
 "V         CVXOPT matrix, n-by-(2*Kj)\n"
 "\n"
 "Ip        CVXOPT integer matrix, |V|-by-1\n"
 "\n"
 "Jp        CVXOPT integer matrix, |V|-by-1\n"
 "\n"
 "j         Integer between 0 and m-1\n";

static PyObject *SCMcolumn
(PyObject *self, PyObject *args)
{
  PyObject *H,*Av,*Ip,*Jp,*V;
  int_t m,n,K;
  int_t i,j,k,l,p,q,r,c;

  if(!PyArg_ParseTuple(args,"OOOOOn",&H,&Av,&V,&Ip,&Jp,&j)) return NULL;

  m = MAT_NCOLS(H);
  n = MAT_NROWS(V);
  K = MAT_NCOLS(V)/2;

  //#pragma omp parallel for shared(m,n,K,Av,Ip,Jp,H,V,j) private(i,l,k,r,c,q,p)
  for (i=j;i<m;i++) {
    p = SP_COL(Av)[i];
    MAT_BUFD(H)[j*m+i] = 0;
    for (l=0;l<SP_COL(Av)[i+1]-p;l++) {
      q = SP_ROW(Av)[p+l];
      r = MAT_BUFI(Ip)[q];
      c = MAT_BUFI(Jp)[q];
      for (k=0;k<K;k++) {
	MAT_BUFD(H)[j*m+i] += SP_VALD(Av)[p+l]*
	  MAT_BUFD(V)[k*n+r]*MAT_BUFD(V)[(K+k)*n+c];
	if (r != c)
	  MAT_BUFD(H)[j*m+i] += SP_VALD(Av)[p+l]*
	    MAT_BUFD(V)[k*n+c]*MAT_BUFD(V)[(K+k)*n+r];
      }
    }
  }

  Py_RETURN_NONE;
}

static PyObject *SCMcolumn2
(PyObject *self, PyObject *args)
{
  PyObject *H,*Av,*Ip,*Jp,*V,*Kl;
  int_t m,n;
  int_t i,j,k,p,q,r,c,r1,c1,pj,pi;
  double alpha,beta;

  if(!PyArg_ParseTuple(args,"OOOOOOn",&H,&Av,&V,&Ip,&Jp,&Kl,&j)) return NULL;

  m = MAT_NCOLS(H);
  n = MAT_NROWS(V);

  for (i=j;i<m;i++) MAT_BUFD(H)[j*m+i] = 0;

  //#pragma omp parallel for shared(m,n,K,Av,Ip,Jp,H,V,j) private(i,l,k,r,c,q,p)
  pj = SP_COL(Av)[j];
  for (p=0;p<SP_COL(Av)[j+1]-pj;p++) {
    alpha = SP_VALD(Av)[pj+p];
    k = SP_ROW(Av)[pj+p];
    r = MAT_BUFI(Ip)[k];
    c = MAT_BUFI(Jp)[k];
    if (r!=c) alpha*=2;
    // look up columns in V
    r = MAT_BUFI(Kl)[r];
    c = MAT_BUFI(Kl)[c];

    for(i=j;i<m;i++) {
      pi = SP_COL(Av)[i];
      for (q=0;q<SP_COL(Av)[i+1]-pi;q++) {
	beta = SP_VALD(Av)[pi+q];
	k = SP_ROW(Av)[pi+q];
	r1 = MAT_BUFI(Ip)[k];
	c1 = MAT_BUFI(Jp)[k];

	MAT_BUFD(H)[j*m+i] += alpha*beta*MAT_BUFD(V)[n*r+r1]*MAT_BUFD(V)[n*c+c1];
	if (r1!=c1)
	  MAT_BUFD(H)[j*m+i] += alpha*beta*MAT_BUFD(V)[n*r+c1]*MAT_BUFD(V)[n*c+r1];
      }
    }
  }

  Py_RETURN_NONE;
}


static char doc_nzcolumns[] =
  "Computes number of nonzero columns in matrices\n"
  "A1,...,Am.\n"
  "\n"
  "nzc = nzcolumns(A)\n"
  "\n"
  "PURPOSE\n"
  "\n"
  "\n"
  "ARGUMENT \n"
  "A        CVXOPT sparse matrix, (n^2)-by-(m+1)\n"
  "\n"
  "RETURNS\n"
  "nzc      CVXOPT integer matrix, m-by-1\n";


static PyObject *nzcolumns
(PyObject *self, PyObject *args)
{
  PyObject *A;
  matrix *Nz;
  int_t m,n,i,j,p,nnz,sum, *tmp;

  if (!PyArg_ParseTuple(args,"O",&A)) return NULL;

  n = (int_t) sqrt((double)SP_NROWS(A));
  m = SP_NCOLS(A)-1;

  Nz = Matrix_New(m,1,INT);
  if (!Nz) return PyErr_NoMemory();

  tmp = malloc(n*sizeof(int_t));
  //tmp = Matrix_New(n,1,INT);
  if (!tmp) return PyErr_NoMemory();

  // erase workspace
  for (i=0;i<n;i++) tmp[i] = 0;

  for (j=0;j<m;j++){
    p = SP_COL(A)[j+1];
    nnz = SP_COL(A)[j+2]-p;
    if (nnz) {
      // Find nonzero cols
      for (i=0;i<nnz;i++) {
	tmp[SP_ROW(A)[p+i] % n] += 1;
	tmp[SP_ROW(A)[p+i] / n] += 1;
      }
      // Count nonzero cols and reset workspace
      MAT_BUFI(Nz)[j] = 0;
      sum = 0;
#pragma omp parallel for shared(tmp,Nz,j,n) private(i) reduction(+:sum)
      for (i=0;i<n;i++) {
	if(tmp[i]) {
	  tmp[i] = 0;
	  sum++;
	}
      }
      MAT_BUFI(Nz)[j] = sum;
    }
  }

  free(tmp);

  return (PyObject*) Nz;
}

static char doc_matperm[] =
  "Computes index permutation.\n"
  "\n"
  "pm,Ns = matperm(nzc,Nmax)\n"
  "\n"
  "PURPOSE\n"
  "\n"
  "\n"
  "ARGUMENT\n"
  "nzc       CVXOPT integer matrix, m-by-1\n"
  "\n"
  "Nmax      Integer\n"
  "\n"
  "RETURNS\n"
  "pm        CVXOPT integer matrix, m-by-1\n"
  "\n"
  "Ns        Integer\n";

static PyObject *matperm
(PyObject *self, PyObject *args)
{
  PyObject *nzc;
  matrix *pm;
  int_t Ns,Nd,m,i,Nmax;

  if (!PyArg_ParseTuple(args,"On",&nzc,&Nmax)) return NULL;
  m = MAT_NROWS(nzc);
  pm = Matrix_New(m,1,INT);
  if (!pm) return PyErr_NoMemory();

  // Check Nmax
  if (Nmax<0) Nmax = 0;

  Ns = 0; Nd = 0;
  for (i=0;i<m;i++){
    if(MAT_BUFI(nzc)[i] > Nmax)
      MAT_BUFI(pm)[Nd++] = i;
    else
      MAT_BUFI(pm)[m-1-Ns++] = i;
  }
  return Py_BuildValue("Nn",pm,Ns);
}

static char doc_toeplitz[] =
  "Generates Toeplitz matrix.\n"
  "\n"
  "T = toeplitz(c[,r])\n"
  "\n"
  "PURPOSE\n"
  "\n"
  "\n"
  "ARGUMENTS\n"
  "c         CVXOPT matrix, m-by-1\n"
  "\n"
  "r         CVXOPT matrix, n-by-1 (OPTIONAL)\n"
  "\n"
  "RETURNS\n"
  "T         CVXOPT matrix\n";

static PyObject *toeplitz
(PyObject *self, PyObject *args, PyObject *kwrds) {

  PyObject *r=NULL, *c=NULL;
  int_t m,n,i,j;
  char *kwlist[] = {"c","r",NULL};

  if (!PyArg_ParseTupleAndKeywords(args,kwrds,"O|O",kwlist,&c,&r)) return NULL;

  if (!Matrix_Check(c))
    return NULL;
  if (r==NULL)
    r = c;
  else if (!Matrix_Check(r))
    return NULL;

  if (MAT_ID(r) != DOUBLE || MAT_ID(c) != DOUBLE)
    return NULL;

  if (MAT_NCOLS(r) > 1 || MAT_NCOLS(c) > 1)
    return NULL;

  m = MAT_NROWS(c);
  n = MAT_NROWS(r);

  // build dense toeplitz matrix, column by column
  matrix *T = Matrix_New(m,n,DOUBLE);
  if (!T) return PyErr_NoMemory();
  for(j=0;j<n;j++) {
    for(i=0;i<(m>j?j:m);i++) {
      MAT_BUFD(T)[j*m+i] = MAT_BUFD(r)[j-i];
    }
    for(i=j;i<m;i++) {
      MAT_BUFD(T)[j*m+i] = MAT_BUFD(c)[i-j];
    }
  }

  return (PyObject*) T;
}

static char doc_robustLS_to_sdp[] =
  "Converts robust least squares problem to SDP.\n"
  "\n"
  "A,bs = robustLS_to_sdp(Alist,b)"
  "\n"
  "PURPOSE\n"
  "Converts the robust least squares problem\n"
  "\n"
  "  minimize e_wc(x)^2\n"
  "\n"
  "  e_wc(x)=sup_{norm(u)<=1} norm((Ab+A1*u1+...+Ap*up)*x - b)\n"
  "\n"
  "to an equivalent SDP of order m+p+1 and with n+2 constraints.\n"
  "\n"
  "ARGUMENTS\n"
  "Alist     List of p+1 CVXOPT matrices of size m-by-n \n"
  "          i.e., Alist = [Ab,A1,...,Ap] \n"
  "\n"
  "b         CVXOPT matrix, m-by-1\n"
  "\n"
  "RETURNS\n"
  "A         CVXOPT sparse matrix\n"
  "\n"
  "bs        CVXOPT sparse matrix\n";

static PyObject *robustLS_to_sdp
(PyObject *self, PyObject *args, PyObject *kwrds) {

  PyObject *Alist,*bt, *Ai;
  spmatrix *A,*b;
  int_t m,n,mp,np,pt,i,j,k,N,nnz=0,ri=0;
  char *kwlist[] = {"Alist","bt",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,kwrds,"OO",kwlist,&Alist,&bt)) return NULL;

  if(!PyList_Check(Alist)) {
    PyErr_SetString(PyExc_TypeError,"Alist must be a list of matrices");
    return NULL;
  }

  // get pt = p + 1
  pt = PyList_Size(Alist);

  // get size of bt
  if(Matrix_Check(bt)){
    m = MAT_NROWS(bt);
    np = MAT_NCOLS(bt);
  }
  else if (SpMatrix_Check(bt)){
    m = SP_NROWS(bt);
    np = SP_NCOLS(bt);
  }
  else {
    PyErr_SetString(PyExc_TypeError,"b must be a vector");
    return NULL;
  }
  if (np!=1) {
    PyErr_SetString(PyExc_TypeError,"b must be a vector");
    return NULL;
  }

  // get n and check A0
  if (!(Ai = PyList_GetItem(Alist,0))) return NULL;
  if (Matrix_Check(Ai)) {
    n = MAT_NCOLS(Ai);
    nnz += m*n;
  }
  else if (SpMatrix_Check(Ai)) {
    n = SP_NCOLS(Ai);
    nnz += SP_NNZ(Ai);
  }
  else {
    PyErr_SetString(PyExc_TypeError,"only spmatrix and matrix types allowed");
    return NULL;
  }

  // check remaining matrices in Alist
  for (i=1;i<pt;i++) {
    if (!(Ai = PyList_GetItem(Alist,i))) return NULL;
    if (Matrix_Check(Ai)) {
      mp = MAT_NROWS(Ai);
      np = MAT_NCOLS(Ai);
      nnz += m*n;
    }
    else if (SpMatrix_Check(Ai)) {
      mp = SP_NROWS(Ai);
      np = SP_NCOLS(Ai);
      nnz += SP_NNZ(Ai);
    }
    else {
      PyErr_SetString(PyExc_TypeError,"only spmatrix and matrix types allowed");
      return NULL;
    }
    if (!(mp==m && np==n)){
      PyErr_SetString(PyExc_TypeError,"matrices in Alist must have same size");
      return NULL;
    }
  }
  nnz += 2*m + pt;

  // generate b
  b = SpMatrix_New(n+2,1,2,DOUBLE);
  if (!b) return PyErr_NoMemory();
  SP_COL(b)[0] = 0;
  SP_VALD(b)[0] = -1;
  SP_ROW(b)[0] = 0;
  SP_VALD(b)[1] = -1;
  SP_ROW(b)[1] = 1;
  SP_COL(b)[1] = 2;

  // generate A
  N = m+pt;
  A = SpMatrix_New(N*N,n+3,nnz,DOUBLE);
  if (!A) return PyErr_NoMemory();

  // build A0
  SP_COL(A)[0] = ri;
  for(i=0;i<m;i++){
    if(SpMatrix_Check(bt)){
      SP_VALD(A)[ri] = -SP_VALD(bt)[i];
      SP_ROW(A)[ri++] = pt+i;
    }
    else{
      SP_VALD(A)[ri] = -MAT_BUFD(bt)[i];
      SP_ROW(A)[ri++] = pt+i;
    }
  }
  for(i=0;i<m;i++) {
    SP_VALD(A)[ri] = 1;
    SP_ROW(A)[ri++] = (N+1)*pt + i*N+i;
  }

  // build A1
  SP_COL(A)[1] = ri;
  for(i=0;i<pt-1;i++){
    SP_VALD(A)[ri] = -1;
    SP_ROW(A)[ri++] = N+1 + i*N+i;
  }

  // build A2
  SP_COL(A)[2] = ri;
  SP_VALD(A)[ri] = -1;
  SP_ROW(A)[ri++] = 0;
  SP_COL(A)[3] = ri;

  // build A3,...
  for(j=0;j<n;j++){
    // generate col. i
    for(i=0;i<pt;i++){
      Ai = PyList_GetItem(Alist,i);
      if(SpMatrix_Check(Ai)) {
	nnz = SP_COL(Ai)[j+1]-SP_COL(Ai)[j];
	for(k=0;k<nnz;k++) {
	  SP_VALD(A)[ri] = -SP_VALD(Ai)[SP_COL(Ai)[j]+k];
	  SP_ROW(A)[ri++] = pt+i*N + SP_ROW(Ai)[SP_COL(Ai)[j]+k];
	}
      }
      else {
	for (k=0;k<m;k++) {
	  SP_VALD(A)[ri] = -MAT_BUFD(Ai)[j*m+k];
	  SP_ROW(A)[ri++] = pt+i*N + k;
	}
      }
    }
    SP_COL(A)[j+4] = ri;
  }

  return Py_BuildValue("NN",A,b);
}

static char doc_phase1_sdp[] =
  "";

static PyObject *phase1_sdp
(PyObject *self, PyObject *args, PyObject *kwrds) {

  matrix *u;
  spmatrix *Ai,*Ao;
  int_t k,i,j,n,m,nnz,nz,col;


  if(!PyArg_ParseTuple(args,"OO",&Ai,&u)) return NULL;
  n = (int_t) sqrt((double)SP_NROWS(Ai));
  m = SP_NCOLS(Ai) - 1;
  nnz = SP_NNZ(Ai) - SP_COL(Ai)[1] + 1 + m + n + 1;

  Ao = SpMatrix_New((n+2)*(n+2),m+2,nnz,DOUBLE);
  if (!Ao) return PyErr_NoMemory();

  // A_0
  SP_VALD(Ao)[0] = 1.0;
  SP_ROW(Ao)[0] = n*(n+2)+n;
  SP_COL(Ao)[0] = 0;
  SP_COL(Ao)[1] = 1;

  // A_i, i=1,..,m
  for (i=1;i<=m;i++){
    k = SP_COL(Ao)[i];
    nz = SP_COL(Ai)[i+1]-SP_COL(Ai)[i]; // nonzeros in Ai
    // copy Ai
    memcpy(SP_VALD(Ao)+k,SP_VALD(Ai)+SP_COL(Ai)[i],nz*sizeof(double));
    // insert -u[i]
    SP_VALD(Ao)[k+nz] = -MAT_BUFD(u)[i-1];
    // update row colptr
    SP_COL(Ao)[i+1] = SP_COL(Ao)[i]+nz+1;
    // generate row indices
    for (j=0;j<nz;j++) {
      col = SP_ROW(Ai)[SP_COL(Ai)[i]+j] / n;
      SP_ROW(Ao)[k+j] = SP_ROW(Ai)[SP_COL(Ai)[i]+j] + col*2;
    }
    SP_ROW(Ao)[k+nz] = n*(n+2)+n;
  }
  // last constraint
  k = SP_COL(Ao)[m+1];
  for (i=0;i<n;i++){
    SP_VALD(Ao)[k+i] = 1.0;
    SP_ROW(Ao)[k+i] = i*(n+2)+i;
  }
  SP_VALD(Ao)[k+n] = 1.0;
  SP_ROW(Ao)[k+n] = (n+2)*(n+2)-1;
  SP_COL(Ao)[m+2] = SP_COL(Ao)[m+1] + n + 1;

  return (PyObject*) Ao;
}


static PyMethodDef misc_functions[] = {

  {"sdpa_readhead", (PyCFunction)sdpa_readhead,
   METH_VARARGS, doc_sdpa_readhead},

  {"sdpa_read", (PyCFunction)sdpa_read,
   METH_VARARGS|METH_KEYWORDS, doc_sdpa_read},

  {"sdpa_write", (PyCFunction)sdpa_write,
   METH_VARARGS|METH_KEYWORDS, doc_sdpa_write},

  {"ind2sub", (PyCFunction)ind2sub,
   METH_VARARGS, doc_ind2sub},

  {"sub2ind", (PyCFunction)sub2ind,
   METH_VARARGS, doc_sub2ind},

  {"scal_diag", (PyCFunction)scal_diag,
   METH_VARARGS, doc_scal_diag},

  {"SCMcolumn", (PyCFunction)SCMcolumn,
   METH_VARARGS, doc_SCMcolumn},

  {"SCMcolumn2", (PyCFunction)SCMcolumn2,
   METH_VARARGS, doc_SCMcolumn},

  {"Av_to_spmatrix", (PyCFunction)Av_to_spmatrix,
   METH_VARARGS|METH_KEYWORDS, doc_Av_to_spmatrix},

  {"nzcolumns", (PyCFunction)nzcolumns,
   METH_VARARGS, doc_nzcolumns},

  {"matperm", (PyCFunction)matperm,
   METH_VARARGS, doc_matperm},

  {"toeplitz", (PyCFunction)toeplitz,
   METH_VARARGS|METH_KEYWORDS, doc_toeplitz},

  {"robustLS_to_sdp", (PyCFunction)robustLS_to_sdp,
   METH_VARARGS|METH_KEYWORDS, doc_robustLS_to_sdp},

  {"phase1_sdp",  (PyCFunction)phase1_sdp,
   METH_VARARGS, doc_phase1_sdp},

  {NULL}  /* Sentinel */
};


#if PY_MAJOR_VERSION >= 3
// Python 3.x
static PyModuleDef misc_module = {
    PyModuleDef_HEAD_INIT,
    "misc",
    misc__doc__,
    -1,
    misc_functions,
    NULL, NULL, NULL, NULL
};
PyMODINIT_FUNC PyInit_misc(void) {
  PyObject *misc_mod;
  misc_mod = PyModule_Create(&misc_module);
  if (misc_mod == NULL)
    return NULL;
  if (import_cvxopt() < 0)
    return NULL;
  return misc_mod;
}
#else
// Python 2.x
PyMODINIT_FUNC initmisc(void) {
  PyObject *m;
  m = Py_InitModule3("misc", misc_functions, misc__doc__);
  if (import_cvxopt() < 0)
    return;
}
#endif
