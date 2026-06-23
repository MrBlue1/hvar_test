package install 
check out the codes to your local folder.
open the terminal, then enter your local folder, set the environment. the instructions are:
1. python -m venv hvar_venv
2. .\scripts\activate
3. pip install -r requirements.txt
4. cd Mutants\softmax
5. python h_test.py
the terminal will show the experiment result from RQ1 to RQ4.

The code in the h_test.py, you can see the line No.999: if __name__=='__main__':
then the execute code line will be wrap up by #Region and #endregion
the RQ1 includes 3 experiments: CM Metrics，Pertubation Test and McNemar significance test, and the exepriments will plot the result data
the RQ2 inclues 3 experiments: Metrics, fingerprint tsne and 3 Cases analysis, and the exepriments will plot the result data
the RQ3 inclues 2 experiments: HVAR pipeline and the Ablation Test, and the exepriments will plot the result data
the RQ4 inclues 2 experiments: Metrics for Mutant Pass & Diagnostic info for Intercepted Mutants, and the exepriments will plot the result data







