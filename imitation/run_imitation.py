from __future__ import annotations
import argparse
import copy
import numpy as np

from simulator.data_loader import load_jobs_from_csv, assign_poisson_arrivals
from simulator.simulator import Simulator
from simulator.schedulers import FIFOScheduler, SJFScheduler
from imitation.imitation_scheduler import train_model, ImitationScheduler
from simulator.models import Cluster


ARRIVAL_RATES = {
    "light": 0.10,
    "moderate": 0.25,
    "heavy": 0.50,
    "extreme": 1.00,
}


def evaluate_policy(name, scheduler, jobs, alpha):
    sim = Simulator(
        num_gpus=10,
        gpu_memory=80.0,
        scheduler=scheduler,
        interference_alpha=alpha,
    )

    metrics = sim.run(copy.deepcopy(jobs))

    print("\n" + "=" * 60)
    print(name)
    print(metrics.summary())
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--train-jobs", type=int, default=6000)
    parser.add_argument("--test-jobs", type=int, default=750)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.0)

    args = parser.parse_args()


    all_jobs = load_jobs_from_csv(args.csv)

    train_jobs = copy.deepcopy(all_jobs[:args.train_jobs])
    test_jobs = copy.deepcopy(
        all_jobs[args.train_jobs:args.train_jobs + args.test_jobs]
    )
    
    # Train the imitation model on the training set (using the expert policy to generate labels)
    cluster = Cluster.create(num_gpus=10, memory_per_gpu=80.0)

    print("Training imitation model...")
    model = train_model(
        train_jobs=train_jobs,
        cluster=cluster,
        epochs=args.epochs,
        lr=1e-3,
    )

    learned_sched = ImitationScheduler(model)


    for label, lam in ARRIVAL_RATES.items():

        print("\n" + "#" * 70)
        print(f"LOAD = {label.upper()}   λ={lam}")
        print("#" * 70)

        test_seed = 42

        jobs_eval = copy.deepcopy(test_jobs)
        jobs_eval = assign_poisson_arrivals(
            jobs_eval,
            arrival_rate=lam,
            seed=test_seed,
        )

        evaluate_policy(
            "FIFO",
            FIFOScheduler(),
            jobs_eval,
            alpha=args.alpha,
        )

        evaluate_policy(
            "SJF",
            SJFScheduler(),
            jobs_eval,
            alpha=args.alpha,
        )

        evaluate_policy(
            "Imitation Neural Scheduler",
            learned_sched,
            jobs_eval,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()