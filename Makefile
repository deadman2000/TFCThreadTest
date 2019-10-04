build:
	rm -f main
	g++ main.cpp -ltensorflow -o main
	./main

docker:
	docker build -t tf_c_test .
	docker run -it --rm tf_c_test
