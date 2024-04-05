#include <iostream>
#include <cstdlib> // For atoi function
#include <cstring> // For strcmp function
#include <cstdarg> // For va_args
#include <chrono>
#include <random>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

// OpenCL variables
cl_int err;
cl_event event;
cl_uint platform_num = 0;
cl_platform_id *platform_list = nullptr;
cl_int platform_id = -1;
cl_uint device_last = 0;
cl_uint device_num = 0;
cl_device_id *device_list = nullptr;
cl_context context;
cl_command_queue *queue_list = nullptr;
cl_uint buffer_size;
cl_mem *buffer_list = nullptr;
double *buffer_host = nullptr;

#define CHECK_ERROR(err, str) check_error_code(err, str, __FILE__, __LINE__);

double get_time();
void crash(const char *fmt, ...);
void check_error_code(cl_int err, const char* operation, const char* file, int line);
void initialize_platform(const char *platform_target);
void deinitialize_platform();
void profile_write_operation(int repeat);
void profile_read_operation(int repeat);
void profile_copy_operation(int repeat, cl_uint src);
void profile_migrate_operation(int repeat, cl_uint dst);

int main(int argc, char **argv){
    // Check if correct number of arguments are provided
    if (argc != 2) {
        printf("Usage: %s <buffer_size>\n", argv[0]);
        return 1;
    }

    buffer_size = std::atoi(argv[1]);

    // Check if buffer_size is greater than 8
    if (buffer_size <= 8) {
        printf("Parameter <buffer_size> must be greater than 8.\n");
        return 1;
    }

    printf("buffer_size: %d\n", buffer_size);

    initialize_platform("Intel(R) OpenCL Graphics");

    printf("Profiling read and write operations with 100 repetitions\n"); 
    profile_write_operation(100);
    profile_read_operation(100);
    printf("\n");

    printf("Profiling copy from 0 to 1; first warm-up, then 100 repetitions\n"); 
    profile_copy_operation(1, 0);
    profile_copy_operation(100, 0);
    profile_copy_operation(1, 0);
    printf("\n");

    printf("Profiling copy from 0 to 1; first update buffers, then 100 repetitions\n"); 
    profile_write_operation(1);
    profile_copy_operation(100, 0);
    printf("\n");

    printf("Profiling ping-pong copy between 0 and 1, 3 times\n");
    profile_copy_operation(1, 1);
    profile_copy_operation(1, 0);
    profile_copy_operation(1, 1);
    profile_copy_operation(1, 0);
    profile_copy_operation(1, 1);
    profile_copy_operation(1, 0);
    printf("\n");

    printf("Profiling migrate operation to 0, first warm-up, then increase repetitions\n");
    profile_migrate_operation(1, 0);
    profile_migrate_operation(1, 0);
    profile_migrate_operation(10, 0);
    profile_migrate_operation(100, 0);

    deinitialize_platform();
    return 0;
}

// Return the current time in seconds since the epoch
double get_time() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

// Utility to print error message
void crash(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "\nEpic fail:\n");
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    fflush(stderr);
    exit(EXIT_FAILURE);
}

// Utility to check OpenCL error codes
void check_error_code(cl_int err, const char* operation, const char* file, int line){
    if(err!=CL_SUCCESS){
        char* str;
        switch(err){
        case CL_SUCCESS:
            str = (char*)"CL_SUCCESS";
            break;
        case CL_DEVICE_NOT_FOUND:
            str = (char*)"CL_DEVICE_NOT_FOUND";
            break;
        case CL_DEVICE_NOT_AVAILABLE:
            str = (char*)"CL_DEVICE_NOT_AVAILABLE";
            break;
        case CL_COMPILER_NOT_AVAILABLE:
            str = (char*)"CL_COMPILER_NOT_AVAILABLE";
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            str = (char*)"CL_MEM_OBJECT_ALLOCATION_FAILURE";
            break;
        case CL_OUT_OF_RESOURCES:
            str = (char*)"CL_OUT_OF_RESOURCES";
            break;
        case CL_OUT_OF_HOST_MEMORY:
            str = (char*)"CL_OUT_OF_HOST_MEMORY";
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            str = (char*)"CL_PROFILING_INFO_NOT_AVAILABLE";
            break;
        case CL_MEM_COPY_OVERLAP:
            str = (char*)"CL_MEM_COPY_OVERLAP";
            break;
        case CL_IMAGE_FORMAT_MISMATCH:
            str = (char*)"CL_IMAGE_FORMAT_MISMATCH";
            break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            str = (char*)"CL_IMAGE_FORMAT_NOT_SUPPORTED";
            break;
        case CL_BUILD_PROGRAM_FAILURE:
            str = (char*)"CL_BUILD_PROGRAM_FAILURE";
            break;
        case CL_MAP_FAILURE:
            str = (char*)"CL_MAP_FAILURE";
            break;
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            str = (char*)"CL_MISALIGNED_SUB_BUFFER_OFFSET";
            break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            str = (char*)"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            break;
        case CL_INVALID_VALUE:
            str = (char*)"CL_INVALID_VALUE";
            break;
        case CL_INVALID_DEVICE_TYPE:
            str = (char*)"CL_INVALID_DEVICE_TYPE";
            break;
        case CL_INVALID_PLATFORM:
            str = (char*)"CL_INVALID_PLATFORM";
            break;
        case CL_INVALID_DEVICE:
            str = (char*)"CL_INVALID_DEVICE";
            break;
        case CL_INVALID_CONTEXT:
            str = (char*)"CL_INVALID_CONTEXT";
            break;
        case CL_INVALID_QUEUE_PROPERTIES:
            str = (char*)"CL_INVALID_QUEUE_PROPERTIES";
            break;
        case CL_INVALID_COMMAND_QUEUE:
            str = (char*)"CL_INVALID_COMMAND_QUEUE";
            break;
        case CL_INVALID_HOST_PTR:
            str = (char*)"CL_INVALID_HOST_PTR";
            break;
        case CL_INVALID_MEM_OBJECT:
            str = (char*)"CL_INVALID_MEM_OBJECT";
            break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            str = (char*)"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            break;
        case CL_INVALID_IMAGE_SIZE:
            str = (char*)"CL_INVALID_IMAGE_SIZE";
            break;
        case CL_INVALID_SAMPLER:
            str = (char*)"CL_INVALID_SAMPLER";
            break;
        case CL_INVALID_BINARY:
            str = (char*)"CL_INVALID_BINARY";
            break;
        case CL_INVALID_BUILD_OPTIONS:
            str = (char*)"CL_INVALID_BUILD_OPTIONS";
            break;
        case CL_INVALID_PROGRAM:
            str = (char*)"CL_INVALID_PROGRAM";
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
            str = (char*)"CL_INVALID_PROGRAM_EXECUTABLE";
            break;
        case CL_INVALID_KERNEL_NAME:
            str = (char*)"CL_INVALID_KERNEL_NAME";
            break;
        case CL_INVALID_KERNEL_DEFINITION:
            str = (char*)"CL_INVALID_KERNEL_DEFINITION";
            break;
        case CL_INVALID_KERNEL:
            str = (char*)"CL_INVALID_KERNEL";
            break;
        case CL_INVALID_ARG_INDEX:
            str = (char*)"CL_INVALID_ARG_INDEX";
            break;
        case CL_INVALID_ARG_VALUE:
            str = (char*)"CL_INVALID_ARG_VALUE";
            break;
        case CL_INVALID_ARG_SIZE:
            str = (char*)"CL_INVALID_ARG_SIZE";
            break;
        case CL_INVALID_KERNEL_ARGS:
            str = (char*)"CL_INVALID_KERNEL_ARGS";
            break;
        case CL_INVALID_WORK_DIMENSION:
            str = (char*)"CL_INVALID_WORK_DIMENSION";
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            str = (char*)"CL_INVALID_WORK_GROUP_SIZE";
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            str = (char*)"CL_INVALID_WORK_ITEM_SIZE";
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            str = (char*)"CL_INVALID_GLOBAL_OFFSET";
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            str = (char*)"CL_INVALID_EVENT_WAIT_LIST";
            break;
        case CL_INVALID_EVENT:
            str = (char*)"CL_INVALID_EVENT";
            break;
        case CL_INVALID_OPERATION:
            str = (char*)"CL_INVALID_OPERATION";
            break;
        case CL_INVALID_GL_OBJECT:
            str = (char*)"CL_INVALID_GL_OBJECT";
            break;
        case CL_INVALID_BUFFER_SIZE:
            str = (char*)"CL_INVALID_BUFFER_SIZE";
            break;
        case CL_INVALID_MIP_LEVEL:
            str = (char*)"CL_INVALID_MIP_LEVEL";
            break;
        case CL_INVALID_GLOBAL_WORK_SIZE:
            str = (char*)"CL_INVALID_GLOBAL_WORK_SIZE";
            break;
        case CL_INVALID_PROPERTY:
            str = (char*)"CL_INVALID_PROPERTY";
            break;
        default:
            str = (char*)"UNKNOWN ERROR";
            break;
        }

        fprintf(stderr, "Error during operation '%s', ", operation);
        fprintf(stderr, "in '%s' on line %d\n", file, line);
        fprintf(stderr, "Error code was \"%s\" (%d)\n", str, err);
        exit(EXIT_FAILURE);
    }
}

void initialize_platform(const char *platform_target) {
    printf("--------------------------------------------------------------------------------\n");
    // Get the number of available platforms
    err = clGetPlatformIDs(0, nullptr, &platform_num);
    CHECK_ERROR(err, "clGetPlatformIDs")

    // Allocate memory to store platform IDs
    platform_list = new cl_platform_id[platform_num];
    err = clGetPlatformIDs(platform_num, platform_list, nullptr);
    CHECK_ERROR(err, "clGetPlatformIDs")
    std::cout << "  clGetPlatformIDs....... ok" << std::endl;

    // Get platform information and search for Intel(R) OpenCL Graphics platform
    char platform_name[1024];
    for (cl_uint i = 0; i < platform_num; ++i) {
        err = clGetPlatformInfo(platform_list[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
        CHECK_ERROR(err, "clGetPlatformInfo")
        std::cout << "    Platform (" << i << ") " << platform_name << std::endl;

        // Check if the current platform matches the target platform
        if (!strcmp(platform_target, platform_name)) {
            platform_id = i;
        }
    }

    // Check if the target platform was found
    if (platform_id < 0) {
        crash("Can't find the required platform");
    }
    std::cout << "    Platform (" << platform_id << ") selected" << std::endl;

    // Get number of devices in selected platform
    err = clGetDeviceIDs(platform_list[platform_id], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_num);
    CHECK_ERROR(err, "clGetDeviceIDs")
    if (device_num < 2) {
        crash("Not enough devices are available");
    }
    std::cout << "  clGetDeviceIDs......... ok" << std::endl;

    // Allocate memory to store device IDs
    device_list = new cl_device_id[device_num];
    err = clGetDeviceIDs(platform_list[platform_id], CL_DEVICE_TYPE_ALL, device_num, device_list, nullptr);
    CHECK_ERROR(err, "clGetDeviceIDs")

    // Print the name of devices in the selected platform
    for (cl_uint i = 0; i < device_num; ++i) {
        err = clGetDeviceInfo(device_list[i], CL_DEVICE_NAME, sizeof(platform_name), platform_name, nullptr);
        CHECK_ERROR(err, "clGetDeviceInfo")
        std::cout << "    Device (" << i << ") " << platform_name << std::endl;
    }

    // Create an OpenCL context
    context = clCreateContext(nullptr, device_num, device_list, nullptr, nullptr, &err);
    CHECK_ERROR(err, "clCreateContext")
    std::cout << "  clCreateContext........ ok" << std::endl;

    // Create command queues for each device
    queue_list = new cl_command_queue[device_num];
    for (cl_uint i = 0; i < device_num; ++i) {
        queue_list[i] = clCreateCommandQueue(context, device_list[i], 0, &err);
        CHECK_ERROR(err, "clCreateCommandQueue")
    }
    std::cout << "  clCreateCommandQueue... ok" << std::endl;

    // Create buffers
    buffer_list = new cl_mem[device_num];
    for (cl_uint i = 0; i < device_num; ++i) {
        buffer_list[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size * sizeof(double), nullptr, &err);
        CHECK_ERROR(err, "clCreateBuffer")
    }

    // Create a host array with random data
    buffer_host = new double[buffer_size];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < buffer_size; ++i) {
        buffer_host[i] = dis(gen);
    }

    // Initialize device buffers with random data
    for (cl_uint i = 0; i < device_num; ++i) {
        err = clEnqueueWriteBuffer(queue_list[i], buffer_list[i], CL_TRUE, 0, buffer_size * sizeof(double), buffer_host, 0, nullptr, nullptr);
        CHECK_ERROR(err, "clEnqueueWriteBuffer")
    }
    printf("  clCreateBuffer......... ok\n");
    printf("--------------------------------------------------------------------------------\n");
}

void deinitialize_platform() {
    if(platform_list) {
        delete[] platform_list;
        platform_list = nullptr;
    }
    if(device_list) {
        delete[] device_list;
        device_list = nullptr;
    }
    if(queue_list) {
        delete[] queue_list;
        queue_list = nullptr;
    }
    if(buffer_list) {
        delete[] buffer_list;
        buffer_list = nullptr;
    }
    if(buffer_host) {
        delete[] buffer_host;
        buffer_host = nullptr;
    }
}

void profile_write_operation(int repeat) {
    double time = 0.0;
    for (cl_uint i = 0; i < device_num; ++i) {
        time = get_time();

        for (int j = 0; j < repeat; ++j) {
            cl_int err = clEnqueueWriteBuffer(queue_list[i], buffer_list[i], CL_TRUE, 0, buffer_size * sizeof(double), buffer_host, 0, nullptr, nullptr);
            CHECK_ERROR(err, "clEnqueueWriteBuffer");
        }

        time = get_time() - time;
        printf(
                "  Hto%d: %3d times %9.3e GB in %9.3e seconds at %8.2f GB/s\n",
                i,
                repeat,
                (double)buffer_size * sizeof(double) / 1e9,
                time,
                (double)buffer_size * sizeof(double) / 1e9 / time * repeat);
    }
}

void profile_read_operation(int repeat) {
    double time = 0.0;
    for (cl_uint i = 0; i < device_num; ++i) {
        time = get_time();

        for (int j = 0; j < repeat; ++j) {
            cl_int err = clEnqueueReadBuffer(queue_list[i], buffer_list[i], CL_TRUE, 0, buffer_size * sizeof(double), buffer_host, 0, nullptr, nullptr);
            CHECK_ERROR(err, "clEnqueueReadBuffer");
        }

        time = get_time() - time;
        printf(
                "  %dtoH: %3d times %9.3e GB in %9.3e seconds at %8.2f GB/s\n",
                i,
                repeat,
                (double)buffer_size * sizeof(double) / 1e9,
                time,
                (double)buffer_size * sizeof(double) / 1e9 / time * repeat);
    }
}

void profile_copy_operation(int repeat, cl_uint src) {
    double time = 0.0;
    for (cl_uint i = 0; i < device_num; ++i) {
        if (i != src) {
            time = get_time();

            for (int j = 0; j < repeat; ++j) {
                err = clEnqueueCopyBuffer(queue_list[i], buffer_list[src], buffer_list[i], 0, 0, buffer_size * sizeof(double), 0, nullptr, &event);
                CHECK_ERROR(err, "clEnqueueCopyBuffer");
                err = clWaitForEvents(1, &event);
                CHECK_ERROR(err, "clWaitForEvents");
            }

            time = get_time() - time;
            printf(
                    "  %dto%d: %3d times %9.3e GB in %9.3e seconds at %8.2f GB/s\n",
                    src,
                    i,
                    repeat,
                    (double)buffer_size * sizeof(double) / 1e9,
                    time,
                    (double)buffer_size * sizeof(double) / 1e9 / time * repeat);
        }
    }
}

void profile_migrate_operation(int repeat, cl_uint dst) {
    double time = 0.0;
    for (cl_uint i = 0; i < device_num; ++i) {
        if (i != dst) {
            time = get_time();

            for (int j = 0; j < repeat; ++j) {
                clEnqueueMigrateMemObjects(queue_list[dst], 1, &buffer_list[i], 0, 0, nullptr, &event);
                CHECK_ERROR(err, "clEnqueueMigrateMemObjects");
                err = clWaitForEvents(1, &event);
                CHECK_ERROR(err, "clWaitForEvents");

                // To repeat the operation, we need to give it back to source 
                clEnqueueMigrateMemObjects(queue_list[i], 1, &buffer_list[dst], 0, 0, nullptr, &event);
                CHECK_ERROR(err, "clEnqueueMigrateMemObjects");
                err = clWaitForEvents(1, &event);
                CHECK_ERROR(err, "clWaitForEvents");
            }

            time = get_time() - time;
            printf(
                    "  %dmg%d: %3d times %9.3e GB in %9.3e seconds at %8.2f GB/s\n",
                    i,
                    dst,
                    repeat,
                    (double)buffer_size * sizeof(double) / 1e9,
                    time / 2,
                    (double)buffer_size * sizeof(double) / 1e9 / time * 2 * repeat );
        }
    }
}
