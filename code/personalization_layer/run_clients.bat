@echo off
set /A NUM_CLIENTS=10
echo Starting %NUM_CLIENTS% clients.
for /l %%i in (1,1,10) do (

    echo "Starting client(cid=%%i) with partition %%i out of %NUM_CLIENTS% clients."
    :: Staggered loading of clients: clients are loaded 8s apart.
    :: At the start, each client loads the entire CIFAR-10 dataset before selecting
    :: their own partition. For a large number of clients this causes a memory usage
    :: spike that can cause client processes to get terminated. 
    :: Staggered loading prevents this.
    timeout 5  
    python client.py --cid=%%i 
)
echo Started %NUM_CLIENTS% clients.