# DEVELOPMENT Log


##### 7dbc058a11936e8b256c4a243a8a508615bbcd39
- Removed cascade codepaths and arguments. 
- Sliced out irrelevant code paths

= Runs first steps in first epoch without crashing


##### 6fd9ac502c6baf85e8bf54fd2b21de14c7af2185

- Undid some work to batchify the model input. Need to focus on train_dataflow first

= Runs without crash 

##### 71afc97b1f4ec61803b37f7e104ffa2c4b6783ea

- Changed get_train_dataflow to batch outputs (batch size set with argument)
- Commit includes code in get_train_dataflow that inspects batched dimensions on first batch (for debugging)

= get_train_dataflow runs without issue. Training should fail as model is not expecting input dimensions and field names ('image' -> 'images')