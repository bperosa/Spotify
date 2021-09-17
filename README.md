# 554project1

I put things into functions, the structure of model.py should be a lot easier to follow. The name == main block is mostly a seperator.

The dict.fromkeys(source, value) just copies the keys from a dictionary like object and resets the values.  that's whats being used.  We can parse the idf and y value files as dictionaries with the words as keys and it should work just fine.

I think I have some semblance of usable code for a full scale meal deal. The testing/ stuff toy loop did what I want, I think. I'm not 100% convinced we have the sparse tensors set up correctly, but that's what makes me uncertain. There is probably a more efficient way to leverage the sparse tensors, but I'm not sure how.

Fun fact  np.isin(a,b) is garbage if a is small and b is large, like we tried...  in is like 50-100x faster, not exaggerating.

## Looking ahead

Mostly as a reminder but we want to make sure we capture stuff like loss at the end of the epochs so we can make plots and get a sense of how many epochs we need. 

TF may automatically do that already... I dont remember. It's been a long day.

