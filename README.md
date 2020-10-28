

https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage



data generalization TODOs
- [x] create new batch sampler with labels
- [x] enable hard negatives
- [x] update LightningDataModule
    - [x] clean up data subpackage
- [x] implement dataset with embeddings
- [x] create embeddingsore
- [x] udpate dataset builder to convert node/item ids
- [x] how to use val/test matrices
- [x] understand pos/neg/blocks setup
    - https://docs.dgl.ai/guide/minibatch-nn.html#guide-minibatch-custom-gnn-module
    - https://docs.dgl.ai/guide/minibatch-node.html#guide-minibatch-node-classification-sampler
- [ ] test end-to-end
- [ ] improve negative sampling


## Distributed Inference

0. linear project everything, save to store
for each layer:
    for each item:
        1. get node ID
        2. sample one neighborhood block
        3. get embeddings for block nodes
