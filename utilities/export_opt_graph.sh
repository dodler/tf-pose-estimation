python run_checkpoint.py --sess=/var/data/tfpose_checkpoints/224_bs64/model_latest-24000 --resize=224x224
bazel-bin/tensorflow/tools/graph_transforms/transform_graph --in_graph=/home/lyan/Documents/tf-pose-estimation/tmp/graph.pb --out_graph=graph_opt.pb --outputs='Openpose/concat_stage7:0' --transforms=' fold_old_batch_norms fold_batch_norms remove_nodes(op=Identity, op=CheckNumerics)'
