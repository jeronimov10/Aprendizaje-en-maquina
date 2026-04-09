#!/bin/bash
# Wait for iter8 to finish, then run P_4JVP and P_5JVP sequentially
while pgrep -f "parte1_iteracion8" > /dev/null; do sleep 15; done
echo "iter8 done at $(date)"

echo "Running P_4JVP..."
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 P_4JVP.ipynb 2>&1 | tee p4_exec.log
echo "P_4JVP done at $(date)"

echo "Running P_5JVP..."
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 P_5JVP.ipynb 2>&1 | tee p5_exec.log
echo "P_5JVP done at $(date)"
