# points
- Does musicGen L > M > S? 
- Why does upper right behave differently?
  - Fast to emerge a diagonal
  - Doesn't commit after long training

# todo
- compare loss of random init VS loss of oracle init.

# 2024/4/4
- 2024_m04_d04@04_58_03_p_fit_batch_dirty_working_tree
  - overfitting to the 1st batch works.  

# 4/6
- every experiment so far doesn't have note duration available to tf piano!!!

# 4/13
- 2024_m04_d11@19_00_40_p_overfit_long
  - overfitting to the 1st batch works.  

# 6/5
I think a 88x88 interpreter may be low-dim and suffer from local minima. Let's try the "free strategy", where the decipher plays arbitrary encodec tokens according to the score. 

# 6/6
Gus: 变与不变， vocabulary within/across song/section. 
LAUI: 1. conference mentioned in email. 2. Science Advance. 
LM mapping: creativity & cognition. 

# 8/19
I cannot find a distribution of permutations that is linear w.r.t. permutation. i.e. the repr of the distribution is linear w.r.t. the permutation matrix. 
My current impl of sample_permutation (a32660c3f3eb42629736b99686bec42cf5668b14) is not linear. 
You need linearity to guarantee that the repr will commit to extremes. 
Now exp results show that they don't commit. The problem could be the lack of linearity. 

_d_sample_sel
no more permutations. Permutation too hard. See what happens with selection. 
