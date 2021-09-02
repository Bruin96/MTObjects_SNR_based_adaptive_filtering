# MTObjects_SNR_based_adaptive_filtering

This repository contains the MTObjects framework as was provided to me by Caroline Haigh. I have further worked on this framework by replacing the Gaussian filter operation with an adaptive filtering operation that varies its smoothing operation based on the SNR value. More specifically, the smoothing strength is varied in such a way as to aim to get all local areas in the image reach a local SNR value equal to the target SNR value.

This project is part of a project for the Rijksuniversiteit Groningen Master's course 'In-Company or Research Internship (CS)'. The first publishing of this repository also represents the final version of the code for this version of the code. There will NOT be support for any issues experienced when trying to use this code, nor is the code verified in any way for accurate business use.

In order to run the code, you need to take the following steps:
- Clone the repository
- Navigate in the command line interface to the folder 'mto\_source\_code'
- Execute the command './compile_\adaptive\_smooth\_lib.sh'
- Execute the command './recompile.sh'
- Run the command 'python3 mto.py \<location of input .fits file \> -out \< out_dir/filename.[fits|png] \>

You can set the inner parameters of the MTObjects framework, including the target SNR value, by using options on the command line. The new parameter introduced in this research project compared to the original code is the '-snr' option, which specifies the target SNR value. The '-snr' option should be followed by a whitespace and a floating point number.
