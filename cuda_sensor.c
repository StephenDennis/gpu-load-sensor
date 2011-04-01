/*  Copyright 2010-2011 Stephen Dennis

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

/* 
TODO:
    - possible expand the arrays in the device info
    - add another var that provides the cuda device count
    - do a dlopen on cuda library
    - add another var that indicates cuda load failed or succeeded
*/

void print_help();

int main(int argc, char** argv)
	{
	struct cudaDeviceProp prop;
	cudaError_t e;
	int count;

	if (argc==2 && (0==strcmp("-help",argv[1])))
		{
        print_help();
		}
	else if (argc==2 && (0==strcmp("-human",argv[1])))
		{
		int i;
		e = cudaGetDeviceCount(&count);
		//printf("GetDeviceCount: error [%d] count [%d]\n",e,count);
		e = cudaGetDeviceProperties(&prop, 0);
		//printf("GetDeviceProperties: error [%d]\n",e);
		for(i=0;i<count;i++)
			{
			if(e!=0)
				{
				printf("cuda.error                [%d][%d]\n",i,e);
				}
			else
				{
				printf("cuda.name                     [%d][%s]\n",i,prop.name);
				printf("cuda.major                    [%d][%d]\n",i,prop.major);
				printf("cuda.minor                    [%d][%d]\n",i,prop.minor);
				printf("cuda.totalGlobalMem           [%d][%d]\n",i,(int)prop.totalGlobalMem);
				printf("cuda.sharedMemPerBlock        [%d][%d]\n",i,(int)prop.sharedMemPerBlock);
				printf("cuda.regsPerBlock             [%d][%d]\n",i,prop.regsPerBlock);
				printf("cuda.warpSize                 [%d][%d]\n",i,prop.warpSize);
				printf("cuda.memPitch                 [%d][%d]\n",i,(int)prop.memPitch);
				printf("cuda.maxThreadsPerBlock       [%d][%d]\n",i,prop.maxThreadsPerBlock);
				printf("cuda.maxThreadsDim            [%d][%d]\n",i,prop.maxThreadsDim[0]); /*print array*/
				printf("cuda.maxGridSize              [%d][%d]\n",i,prop.maxGridSize[0]); /*print array*/
				printf("cuda.totalConstMem            [%d][%d]\n",i,(int)prop.totalConstMem);
				printf("cuda.clockRate                [%d][%d]\n",i,prop.clockRate);
				printf("cuda.textureAlignment         [%d][%d]\n",i,(int)prop.textureAlignment);
				printf("cuda.deviceOverlap            [%d][%d]\n",i,prop.deviceOverlap);
				printf("cuda.multiProcessorCount      [%d][%d]\n",i,prop.multiProcessorCount);
				printf("cuda.kernelExecTimeoutEnabled [%d][%d]\n",i,prop.kernelExecTimeoutEnabled);
				printf("cuda.integrated               [%d][%d]\n",i,prop.integrated);
				printf("cuda.canMapHostMemory         [%d][%d]\n",i,prop.canMapHostMemory);
				printf("cuda.computeNode              [%d][%d]\n",i,prop.computeMode);
				}
			}
		}
	else
		{
		char s[1024];
        char h[1024];
        *h=0;
        gethostname(h,sizeof(h));
		while (fgets(s,sizeof(s),stdin) != 0)
			{
			int i;
			if (s && (0==strncmp("quit",s,4)))
				break;
			printf("begin\n");
			e = cudaGetDeviceCount(&count);
			e = cudaGetDeviceProperties(&prop, 0);
			for(i=0;i<count;i++)
				{
				printf("%s:cuda.%d.name:%s\n",h,i,prop.name);
				printf("%s:cuda.%d.major:%d\n",h,i,prop.major);
				printf("%s:cuda.%d.minor:%d\n",h,i,prop.minor);
				printf("%s:cuda.%d.totalGlobalMem:%d\n",h,i,(int)prop.totalGlobalMem);
				printf("%s:cuda.%d.sharedMemPerBlock:%d\n",h,i,(int)prop.sharedMemPerBlock);
				printf("%s:cuda.%d.regsPerBlock:%d\n",h,i,prop.regsPerBlock);
				printf("%s:cuda.%d.warpSize:%d\n",h,i,prop.warpSize);
				printf("%s:cuda.%d.memPitch:%d\n",h,i,(int)prop.memPitch);
				printf("%s:cuda.%d.maxThreadsPerBlock:%d\n",h,i,prop.maxThreadsPerBlock);
				printf("%s:cuda.%d.maxThreadsDim:%d\n",h,i,prop.maxThreadsDim[0]); /*maybe todo print array*/
				printf("%s:cuda.%d.maxGridSize:%d\n",h,i,prop.maxGridSize[0]); /*maybe todo print array*/
				printf("%s:cuda.%d.totalConstMem:%d\n",h,i,(int)prop.totalConstMem);
				printf("%s:cuda.%d.clockRate:%d\n",h,i,prop.clockRate);
				printf("%s:cuda.%d.textureAlignment:%d\n",h,i,(int)prop.textureAlignment);
				printf("%s:cuda.%d.deviceOverlap:%d\n",h,i,prop.deviceOverlap);
				printf("%s:cuda.%d.multiProcessorCount:%d\n",h,i,prop.multiProcessorCount);
				printf("%s:cuda.%d.kernelExecTimeoutEnabled:%d\n",h,i,prop.kernelExecTimeoutEnabled);
				printf("%s:cuda.%d.integrated:%d\n",h,i,prop.integrated);
				printf("%s:cuda.%d.canMapHostMemory:%d\n",h,i,prop.canMapHostMemory);
				printf("%s:cuda.%d.computeNode:%d\n",h,i,prop.computeMode);
				}
			printf("end\n");
			}
		}
	}


void print_help()
    {

printf(
"This program is a Grid Engine load sensor\n"
"\n"
"To configure your grid engine installation to accept load values from this program\n"
"you must add a set of values to your complex.  See the man page for complex for\n"
"more information.\n"
"\n"
"The following is a bash script which you can run to configure your complex.\n"
"\n"
"You can have more than one cuda device on a computer.  The script takes one \n"
"parameter which is the number of cuda devices, with a default of one.\n"
"\n"
"This loadsensor needs to be run on each of your cuda hosts.  \n"
"Currently this sensor does not dynamically load so you should only run it on\n"
"hosts with cuda installed or it will just fail to run.  You configure the\n"
"hosts with qconf -mconf, and adding or modifying a load_sensor value or entry.\n"
"\n"
"See the load_sensor man page. \n"
" \n"
"#!/bin/sh\n"
"\n"
"if [ \"$1\" = \"\" ]; then\n"
"    echo Assuming 1 cuda device per compute node\n"
"    NUM_CUDA=0\n"
"elif\n"
"    NUM_CUDA=$1\n"
"fi\n"
"\n"
"add_int_load_complex ()\n"
"{\n"
"qconf -sc > /tmp/complex_dump\n"
"printf \"${1}\t${1}\tINT\t<=\tYES\tNO\t0\t0\n\" >> /tmp/complex_dump\n"
"qconf -Mc /tmp/complex_dump\n"
"}\n"
"\n"
"add_string_complex ()\n"
"{\n"
"qconf -sc > /tmp/complex_dump\n"
"printf \"${1}\t${1}\tRESTRING\t==\tYES\tNO\tNONE\t0\n\" >> /tmp/complex_dump\n"
"qconf -Mc /tmp/complex_dump\n"
"}\n"
"\n"
"for i in $NUM_CUDA ; do \n"
"    add_string_complex cuda.$i.name\n"
"    add_int_load_complex cuda.$i.major\n"
"    add_int_load_complex cuda.$i.minor\n"
"    add_int_load_complex cuda.$i.totalGlobalMem\n"
"    add_int_load_complex cuda.$i.sharedMemPerBlock\n"
"    add_int_load_complex cuda.$i.regsPerBlock\n"
"    add_int_load_complex cuda.$i.warpSize\n"
"    add_int_load_complex cuda.$i.memPitch\n"
"    add_int_load_complex cuda.$i.maxThreadsPerBlock\n"
"    add_int_load_complex cuda.$i.maxThreadsDim\n"
"    add_int_load_complex cuda.$i.maxGridSize\n"
"    add_int_load_complex cuda.$i.totalConstMem\n"
"    add_int_load_complex cuda.$i.clockRate\n"
"    add_int_load_complex cuda.$i.textureAlignment\n"
"    add_int_load_complex cuda.$i.deviceOverlap\n"
"    add_int_load_complex cuda.$i.multiProcessorCount\n"
"    add_int_load_complex cuda.$i.kernelExecTimeoutEnabled\n"
"    add_int_load_complex cuda.$i.integrated\n"
"    add_int_load_complex cuda.$i.canMapHostMemory\n"
"    add_int_load_complex cuda.$i.computeNode\n"
"done \n"
);

    }
