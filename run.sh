
c++ $1.cpp -lboost_system -lboost_filesystem -lboost_random `pkg-config --cflags opencv` `pkg-config --libs opencv` -o run -Iann/include -L/usr/local/lib/ -lANN
./run
