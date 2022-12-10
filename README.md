# Operator Splitting Value Iteration
This is the code for experiments in the paper "Operator Splitting Value Iteration" at NeurIPS2022. 

## To reproduce all the plots
Simply run the runner.sh script in bash:

```
sh ./runner.sh
```

## To run an OSVI experiment

You can use exp_pe_vector.py or exp_control_vector.py for the PE and control problems along a config file. For example:

```
python ./exp_pe_vector.py <./config/configfilename.yaml> ALL
python ./exp_control_vector.py <./config/configfilename.yaml> ALL
```

## To run an OSDyna experiment

You can use exp_pe_sample.py or exp_control_sample.py for the PE and control problems along a config file. For example:
```
python ./exp_pe_sample.py <./config/configfilename.yaml> ALL --num_trials 20
python ./exp_control_sample.py <./config/configfilename.yaml> ALL --num_trials 20
```

