#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <tensorflow/c/c_api.h>


void parse_output(char *buf, size_t bufsize, const char cmd[])
{
    FILE *fp;

    // add dynamic allocation here
    memset(buf, 0, bufsize);

    if ((fp = popen(cmd, "r")) == NULL) {
        printf("Error opening pipe!\n");
        exit(-__LINE__);
    }

    // worst speed ever. And strlen is called twice...
    while (fgets(&buf[strlen(buf)], bufsize - strlen(buf), fp) != NULL);

    if(pclose(fp))  {
        printf("Command not found or exited with error status\n");
        exit(-__LINE__);
    }
}

void display_threads() {
    pid_t pid;
    char cmd[256];
    char buf[256];
    long num;

    pid = getpid();
    printf("PID: %d\n", pid);

    sprintf(cmd, "cat /proc/%d/status |grep Threads", pid);

    parse_output(buf, sizeof(buf), cmd);
    printf("%s", buf);
}



constexpr char kSavedModelTagServe[] = "serve";

static void FloatDeallocator(void* data, size_t, void* arg) {
	delete[] static_cast<float*>(data);
}

void work()
{
	TF_SessionOptions* opt = TF_NewSessionOptions();
	TF_Buffer* run_options = TF_NewBufferFromString("", 0);
	const char * model_path = "model/";
	const char* tags[] = { kSavedModelTagServe };
	TF_Graph* graph = TF_NewGraph();
	TF_Buffer* metagraph = TF_NewBuffer();
	TF_Status* s = TF_NewStatus();
	TF_Session* session = TF_LoadSessionFromSavedModel(opt, run_options, model_path, tags, 1, graph, metagraph, s);

	TF_DeleteBuffer(run_options);
	TF_DeleteSessionOptions(opt);

	TF_Output sp = { TF_GraphOperationByName(graph, "sp") };
	TF_Output fuel = { TF_GraphOperationByName(graph, "fuel") };
	TF_Output softmax = { TF_GraphOperationByName(graph, "softmax_tensor") };

	int NumInputs = 2;
	TF_Output Input[] = { sp, fuel };

	int NumOutputs = 1;
	TF_Output Output[] = { softmax };

	const int num_bytes = 96 * sizeof(float);
	int64_t dims[] = { 1, 96 };

	//while (true)
	for (int c = 0; c < 1; ++c)
	{
		float* values0 = new float[96];
		for (int i = 0; i < 96; ++i)values0[i] = 0.f;
		TF_Tensor* t0 = TF_NewTensor(TF_FLOAT, dims, 2, values0, num_bytes, &FloatDeallocator, nullptr);

		float* values1 = new float[96];
		for (int i = 0; i < 96; ++i)values1[i] = 0.f;
		TF_Tensor* t1 = TF_NewTensor(TF_FLOAT, dims, 2, values1, num_bytes, &FloatDeallocator, nullptr);

		TF_Tensor* InputValues[] = { t0, t1 };
		TF_Tensor* OutputValues[] = { nullptr };

		TF_SessionRun(session, nullptr, Input, InputValues, NumInputs,
			Output, OutputValues, NumOutputs,
			nullptr, 0, nullptr, s);
		float* result = reinterpret_cast<float*>(TF_TensorData(OutputValues[0]));

		TF_DeleteTensor(t0);
		TF_DeleteTensor(t1);
		TF_DeleteTensor(OutputValues[0]);
	}

	TF_DeleteBuffer(metagraph);
	//TF_CloseSession(session, s);
	TF_DeleteSession(session, s);
	TF_DeleteGraph(graph);
	TF_DeleteStatus(s);
}


int main() {
	printf("Hello from TensorFlow C library version %s\n", TF_Version());

	while (true)
	{
		work();

		display_threads();
		printf("Enter q to break or press Enter to repeat\n");
		if (getchar() == 'q') break;
	}

	return 0;
}
