import pandas as pd
import numpy as np
import dt

if __name__ == "__main__":
    task = "flights-regress"
    y_str = dt.const.Y_COLUMN[task]

    train = dt.dbsource.get_train(task)
    print("Train set:")
    print(train)
    print()

    test = dt.dbsource.get_test(task)
    print("Test set:")
    print(test)
    print()

    train_losses, train_stats = dt.pipeline.pipeline_train_py(train, test, 1, task) 

    print("Train losses:")
    print(train_losses)
    print()

    print("Train stats:")
    print(train_stats)
    print()

    slices, sliceline_stats = dt.pipeline.pipeline_sliceline_py(train, train_losses, 1.0, 3, 20, 5, task)

    print("Slices:")
    print(slices)
    print()

    print("Sliceline stats:")
    print(sliceline_stats)
    print()

    explore_scale = len(train) / sum(sliceline_stats['sizes'])
    print("exploration scale:", explore_scale)

    costs = [1.0 for _ in range(dt.const.SOURCES[task]['n'])]
    query_counts = [100 for slice_ in slices]
    gt_stats = utils.construct_stats_table(slices, task)
    dt_results = dt.pipeline.pipeline_dt_py(slices, costs, query_counts, "random", 
        train, ["random", "exploreexploit",    "ratiocoll"], explore_scale, 
        gt_stats, "flights-regress")

    print("DT results:")
    print(dt_results)