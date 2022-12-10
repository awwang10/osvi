# Create output directory
mkdir output

# PE - VECTOR
echo "#### PE-Maze-Smoothed ####"
python ./exp_pe_vector.py ./config/config_vector_maze_smoothed.yaml ALL

echo "#### PE-Garnet-Smoothed ####"
python ./exp_pe_vector.py ./config/config_vector_garnet_smoothed.yaml ALL

echo "#### PE-Cliffwalk-Smoothed ####"
python ./exp_pe_vector.py ./config/config_vector_cliffwalk_smoothed.yaml ALL

echo "#### PE-Maze-Identity ####"
python ./exp_pe_vector.py ./config/config_vector_maze_identity.yaml ALL

echo "#### PE-Garnet-Identity ####"
python ./exp_pe_vector.py ./config/config_vector_garnet_identity.yaml ALL

echo "#### PE-Cliffwalk-Identity ####"
python ./exp_pe_vector.py ./config/config_vector_cliffwalk_identity.yaml ALL



# Control - VECTOR
echo "#### Control-Maze-Smoothed ####"
python ./exp_control_vector.py ./config/config_vector_maze_smoothed.yaml ALL

echo "#### Control-Garnet-Smoothed ####"
python ./exp_control_vector.py ./config/config_vector_garnet_smoothed.yaml ALL

echo "#### Controll-Cliffwalk-Smoothed ####"
python ./exp_control_vector.py ./config/config_vector_cliffwalk_smoothed.yaml ALL


echo "#### Control-Maze-Identity ####"
python ./exp_control_vector.py ./config/config_vector_maze_identity.yaml ALL

echo "#### Control-Garnet-Identity ####"
python ./exp_control_vector.py ./config/config_vector_garnet_identity.yaml ALL

echo "#### Control-Cliffwalk-Identity ####"
python ./exp_control_vector.py ./config/config_vector_cliffwalk_identity.yaml ALL


# PE - SAMPLE
echo "#### PE_SAMPLE_RESCALED ####"
python ./exp_pe_sample.py ./config/config_samples_cliffwalk_rescaled_linear.yaml ALL --num_trials 20

echo "#### PE_SAMPLE_CONSTANT_DELAY ####"
python ./exp_pe_sample.py ./config/config_samples_cliffwalk_constantdelay.yaml ALL --num_trials 20


# Control - SAMPLE
echo "#### CONTROL_SAMPLE_RESCALED ####"
python ./exp_control_sample.py ./config/config_samples_cliffwalk_rescaled_linear.yaml ALL --num_trials 20

echo "#### CONTROL_SAMPLE_CONSTANT_DELAY ####"
python ./exp_control_sample.py ./config/config_samples_cliffwalk_constantdelay.yaml ALL --num_trials 20


## Moving Data
mkdir output/vector_uniform_final
mkdir output/vector_identity_final
mkdir output/sample_pe_final
mkdir output/sample_control_final
mkdir output/main_control_final
#
## Copy data from output directories to plotting directory
cp -r output/8da6_maze33 output/vector_uniform_final/8da6_maze33
cp -r output/0d2f_garnet output/vector_uniform_final/0d2f_garnet
cp -r output/2145_cliffwalk output/vector_uniform_final/2145_cliffwalk

cp -r output/dcb2_maze33 output/vector_identity_final/dcb2_maze33
cp -r output/7e5e_cliffwalk output/vector_identity_final/7e5e_cliffwalk
cp -r output/c130_garnet output/vector_identity_final/c130_garnet

cp -r output/799a_cliffwalk/exp_pe_sample output/sample_pe_final/constant
cp -r output/799a_cliffwalk/exp_control_sample output/sample_control_final/constant_withdelay

cp -r output/1f97_cliffwalk/exp_pe_sample output/sample_pe_final/rescaled_linear
cp -r output/1f97_cliffwalk/exp_control_sample output/sample_control_final/rescaled_linear

cp -r output/2145_cliffwalk/exp_control_vector0 output/main_control_final/vector
cp -r output/799a_cliffwalk/exp_control_sample output/main_control_final/sample

#Plotting

#Figure 5a
python ./plotter_fig5a.py ./config/config_vector_maze_smoothed.yaml ./config/config_vector_garnet_smoothed.yaml ./config/config_vector_cliffwalk_smoothed.yaml 8da6_maze33 0d2f_garnet 2145_cliffwalk --num_trials 20 --exp_dir=vector_uniform_final

#Figure 5b
python ./plotter_fig5b.py ./config/config_vector_maze_smoothed.yaml ./config/config_vector_garnet_smoothed.yaml ./config/config_vector_cliffwalk_smoothed.yaml 8da6_maze33 0d2f_garnet 2145_cliffwalk --num_trials 20 --exp_dir=vector_uniform_final

#Figure 6
python ./plotter_fig6a_fig7a.py ./config/config_vector_maze_smoothed.yaml ./config/config_vector_garnet_smoothed.yaml ./config/config_vector_cliffwalk_smoothed.yaml 8da6_maze33 0d2f_garnet 2145_cliffwalk --num_trials 20 --exp_dir=vector_uniform_final --output_file="OSVI-PE-VariedLambda-Smooth-Maze-Garnet(runs=100, S=50, A=4, bp=3, br=5)-ModifiedCliffwalk.pdf"
python ./plotter_fig6a_fig7a.py ./config/config_vector_maze_identity.yaml ./config/config_vector_garnet_identity.yaml ./config/config_vector_cliffwalk_identity.yaml dcb2_maze33 c130_garnet 7e5e_cliffwalk --num_trials 20 --exp_dir=vector_identity_final --output_file="OSVI-PE-VariedLambda-SelfLoop-Maze-Garnet(runs=100, S=50, A=4, bp=3, br=5)-ModifedCliffwalk.pdf"

#Figure 7
python ./plotter_fig6b_fig7b.py ./config/config_vector_maze_smoothed.yaml ./config/config_vector_garnet_smoothed.yaml ./config/config_vector_cliffwalk_smoothed.yaml 8da6_maze33 0d2f_garnet 2145_cliffwalk --num_trials 20 --exp_dir=vector_uniform_final --output_file="OSVI-Control-VariedLambda-Smooth-Maze-Garnet(runs=100, S=50, A=4, bp=3, br=5)-ModifiedCliffwalk.pdf"
python ./plotter_fig6b_fig7b.py ./config/config_vector_maze_identity.yaml ./config/config_vector_garnet_identity.yaml ./config/config_vector_cliffwalk_identity.yaml dcb2_maze33 c130_garnet 7e5e_cliffwalk --num_trials 20 --exp_dir=vector_identity_final --output_file="OSVI-Control-VariedLambda-SelfLoop-Maze-Garnet(runs=100, S=50, A=4, bp=3, br=5)-ModifedCliffwalk.pdf"

#Figure 8
python ./plotter_fig8.py ./config/config_samples_cliffwalk_constantdelay.yaml ./config/config_samples_cliffwalk_rescaled_linear.yaml constant rescaled_linear --num_trials 20 --exp_dir=sample_pe_final --plot_every=100
python ./plotter_fig9.py ./config/config_samples_cliffwalk_constantdelay.yaml ./config/config_samples_cliffwalk_rescaled_linear.yaml constant_withdelay rescaled_linear --num_trials 20 --plot_every 1000 --exp_dir=sample_control_final

# Body Plot
python ./plotter_main.py ./config/config_vector_cliffwalk_smoothed.yaml ./config/config_samples_cliffwalk_constantdelay.yaml vector sample --num_trials 20 --plot_every 1000 --exp_dir=main_control_final --num_alphas=3