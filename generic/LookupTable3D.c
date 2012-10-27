#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LookupTable3D.c"
#else

static int nn_(LookupTable3D_updateOutput)(lua_State *L)
{
  THTensor * input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int inputFrameSize = luaT_getfieldcheckint(L, 1, "inputFrameSize");

  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  THTensor * size = luaT_getfieldcheckudata(L, 1, "size", luaT_checktypename2id(L, "torch.LongStorage"));

  int nOutputFrame, nOutputFrameSize;
  long j,k;
  
  luaL_argcheck(L, input->nDimension == 1, 2, "1D tensor expected");
  luaL_argcheck(L, input->size[0] >= inputFrameSize, 2, "input is too small");

  input = THTensor_(newContiguous)(input);
  output = THTensor_(newContiguous)(output);
  weight = THTensor_(newContiguous)(weight);

  int wfsz = size->size[1];
  nOutputFrame = (int)ceil((real)(input->size[0]-(inputFrameSize-1))/dW);
  //printf("(%ld - (%d-1) ) / %d = %f\n", input->size[0], inputFrameSize, dW, ((real)(input->size[0]-(inputFrameSize-1))/dW));
  size->size[0] = nOutputFrame;
  nOutputFrameSize = wfsz*inputFrameSize;
  //printf("(%d,%d,%d)\n",nOutputFrame,nOutputFrameSize,wfsz);

  THTensor_(resize2d)(output,
                      nOutputFrame,
                      nOutputFrameSize);

  /* define pointer to data elements */
  real * ptr_data = THTensor_(data)(output);
  /* define pointer to weights */
  real * ptr_weight = THTensor_(data)(weight);
  /* define pointer to input */
  real * ptr_input = THTensor_(data)(input);

  int i=0;
  
#pragma omp parallel for private(k)
  for(k = 0; k<nOutputFrame; k++)
  {
    i=k*dW; // get the right input index
    for (j=0; j<inputFrameSize; j++)
    {
      memcpy(ptr_data+(k*nOutputFrameSize+j*wfsz), ptr_weight+(((int)ptr_input[i+j]-1)*wfsz), sizeof(real)*wfsz);
    } 
  }
  
  THTensor_(free)(input);

  return 1;
}

static int nn_(LookupTable3D_accGradParameters)(lua_State *L)
{
  THTensor * input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor * gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  real scale = luaL_optnumber(L, 4, 1);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int inputFrameSize = luaT_getfieldcheckint(L, 1, "inputFrameSize");
//  long nInputFrame = input->size[0];
  long nOutputFrame = gradOutput->size[0];
  long nOutputFrameSize = gradOutput->size[1];

  THTensor * gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor * size = luaT_getfieldcheckudata(L, 1, "size", luaT_checktypename2id(L, "torch.LongStorage"));

  int wfsz = size->size[1];
  long k,j,index;

  input = THTensor_(newContiguous)(input);
  gradWeight = THTensor_(newContiguous)(gradWeight);
  /* define pointer to weights */
  real * ptr_gradWeight = THTensor_(data)(gradWeight);
  /* define pointer to input */
  real * ptr_input = THTensor_(data)(input);
  /* define pointer to gradients */
  real * ptr_gradOutput = THTensor_(data)(gradOutput);

  // push the table on the stack
  luaT_getfieldchecktable(L, 1, "inputs");
  
  int ind=0,i=0;
#pragma omp parallel for private(k)
  for (k=0; k<nOutputFrame; k++)
  {
    ind=k*dW; // get the right input index
    for(j=0; j<inputFrameSize; j++)
    {
      index = ptr_input[ind+j]-1;
      lua_pushnumber(L, index+1); /* Push the table index */
      lua_pushboolean(L, 1); /* Push the true value */
 //     luaT_stackdump(L);
      lua_rawset(L, -3);      /* Stores the pair in the table */
      for(i=0; i<wfsz; i++)
      {
        ptr_gradWeight[index*wfsz+i] += scale*ptr_gradOutput[k*nOutputFrameSize+j*wfsz+i];
      }
    }
  }

  THTensor_(free)(input);

  return 0;
}

static int nn_(LookupTable3D_accUpdateGradParameters)(lua_State *L)
{
  THTensor * input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor * gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  real lr = luaL_optnumber(L, 4, 1);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int inputFrameSize = luaT_getfieldcheckint(L, 1, "inputFrameSize");
//  long nInputFrame = input->size[0];
  long nOutputFrame = gradOutput->size[0];
  long nOutputFrameSize = gradOutput->size[1];

  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor * size = luaT_getfieldcheckudata(L, 1, "size", luaT_checktypename2id(L, "torch.LongStorage"));

  int wfsz = size->size[1];
  long k,j,index;

  input = THTensor_(newContiguous)(input);
  weight = THTensor_(newContiguous)(weight);
  /* define pointer to weights */
  real * ptr_weight = THTensor_(data)(weight);
  /* define pointer to input */
  real * ptr_input = THTensor_(data)(input);
  /* define pointer to gradients */
  real * ptr_gradOutput = THTensor_(data)(gradOutput);

  int ind=0,i=0;
#pragma omp parallel for private(k)
  for (k=0; k<nOutputFrame; k++)
  {
    ind=k*dW; // get the right input index
    for(j=0; j<inputFrameSize; j++)
    {
      index = ptr_input[ind+j]-1;
      for(i=0; i<wfsz; i++)
      {
        ptr_weight[index*wfsz+i] -= lr*ptr_gradOutput[k*nOutputFrameSize+j*wfsz+i];
      }
    }
  }

  THTensor_(free)(input);

  return 0;
}

static const struct luaL_Reg nn_(LookupTable3D__) [] = {
  {"LookupTable3D_updateOutput", nn_(LookupTable3D_updateOutput)},
  {"LookupTable3D_accGradParameters", nn_(LookupTable3D_accGradParameters)},
  {"LookupTable3D_accUpdateGradParameters", nn_(LookupTable3D_accUpdateGradParameters)},
  {NULL, NULL}
};

static void nn_(LookupTable3D_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(LookupTable3D__), "nn");
  lua_pop(L,1);
}

#endif
