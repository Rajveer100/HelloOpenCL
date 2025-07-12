//
//  main.c
//  HelloOpenCL
//
//  Created by Rajveer Singh on 12/07/25.
//

#include <stdio.h>
#include <stdlib.h>

#include <OpenCL/opencl.h>

#include <CoreFoundation/CoreFoundation.h>

#define NUM_VALUES 1024

char* getBundleResourcePath(const char* filename) {
  CFBundleRef mainBundle = CFBundleGetMainBundle();
  CFStringRef cfFilename = CFStringCreateWithCString(NULL, filename, kCFStringEncodingUTF8);
  CFURLRef fileURL = CFBundleCopyResourceURL(mainBundle, cfFilename, NULL, NULL);
  
  if (fileURL == NULL) {
    CFRelease(cfFilename);
    return NULL;
  }
  
  char* path = malloc(PATH_MAX);
  Boolean result = CFURLGetFileSystemRepresentation(fileURL, TRUE, (UInt8*)path, PATH_MAX);
  
  CFRelease(fileURL);
  CFRelease(cfFilename);
  
  if (!result) {
    free(path);
    return NULL;
  }
  
  return path;
}

static int validate(cl_float* input, cl_float* output) {
  for (int i = 0; i < NUM_VALUES; i++) {
    if (output[i] != (input[i] * input[i])) {
      fprintf(stdout, "Error: Element %d did not match expected output.\n", i);
      fprintf(stdout, "Saw %1.4f, expected %1.4f\n", output[i], input[i] * input[i]);
      fflush(stdout);
      return 0;
    }
  }
  return 1;
}

int main(int argc, const char* argv[]) {
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  
  // Get platform
  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS) {
    printf("Error getting platform: %d\n", err);
    return 1;
  }
  
  // Get device (try GPU first, then CPU)
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) {
    printf("No GPU found, trying CPU...\n");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
      printf("No OpenCL devices found\n");
      return 1;
    }
  }
  
  // Print device name
  char device_name[256];
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
  printf("Using device: %s\n", device_name);
  
  // Create context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    printf("Error creating context: %d\n", err);
    return 1;
  }
  
  // Create command queue
  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS) {
    printf("Error creating command queue: %d\n", err);
    clReleaseContext(context);
    return 1;
  }
  
  // Read the kernel source from the file
  const char* kernel_file = "kernel.cl";
  char* kernel_source = getBundleResourcePath(kernel_file);
  
  if (kernel_source == NULL) {
    printf("Could not find %s in bundle\n", kernel_file);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  // Create program
  program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &err);
  if (err != CL_SUCCESS) {
    printf("Error creating program: %d\n", err);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  // Build program
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error building program: %d\n", err);
    
    // Print build log
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char* log = malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("Build log:\n%s\n", log);
    free(log);
    
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  // Create kernel
  kernel = clCreateKernel(program, "square", &err);
  if (err != CL_SUCCESS) {
    printf("Error creating kernel: %d\n", err);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  // Create test data
  float* test_in = malloc(sizeof(float) * NUM_VALUES);
  float* test_out = malloc(sizeof(float) * NUM_VALUES);
  
  for (int i = 0; i < NUM_VALUES; i++) {
    test_in[i] = (float)i;
  }
  
  // Create OpenCL buffers
  cl_mem mem_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * NUM_VALUES, test_in, &err);
  if (err != CL_SUCCESS) {
    printf("Error creating input buffer: %d\n", err);
    goto cleanup;
  }
  
  cl_mem mem_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  sizeof(float) * NUM_VALUES, NULL, &err);
  if (err != CL_SUCCESS) {
    printf("Error creating output buffer: %d\n", err);
    clReleaseMemObject(mem_in);
    goto cleanup;
  }
  
  // Set kernel arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_in);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_out);
  if (err != CL_SUCCESS) {
    printf("Error setting kernel arguments: %d\n", err);
    clReleaseMemObject(mem_in);
    clReleaseMemObject(mem_out);
    goto cleanup;
  }
  
  // Execute kernel
  size_t global_size = NUM_VALUES;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error executing kernel: %d\n", err);
    clReleaseMemObject(mem_in);
    clReleaseMemObject(mem_out);
    goto cleanup;
  }
  
  // Read results
  err = clEnqueueReadBuffer(queue, mem_out, CL_TRUE, 0, sizeof(float) * NUM_VALUES, test_out, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error reading results: %d\n", err);
    clReleaseMemObject(mem_in);
    clReleaseMemObject(mem_out);
    goto cleanup;
  }
  
  // Validate results
  if (validate(test_in, test_out)) {
    printf("All values were properly squared!\n");
  }
  
  // Cleanup
  clReleaseMemObject(mem_in);
  clReleaseMemObject(mem_out);
  
cleanup:
  free(test_in);
  free(test_out);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  
  return 0;
}
