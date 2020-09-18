

https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage



data generalization TODOs
- [x] create new batch sampler with labels
- [x] enable hard negatives
- [x] update LightningDataModule
    - [x] clean up data subpackage
- [x] implement dataset with embeddings
- [x] create embeddingsore
- [x] udpate dataset builder to convert node/item ids
- [ ] how to use val/test matrices
- [ ] understand pos/neg/blocks setup
- [ ] how to use precomputed embeddings for each node which dont fit in memory?
    - [ ] update dataset, sampler, collater, linear projector to match
