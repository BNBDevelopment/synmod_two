"""Master pipeline"""


import argparse
from distutils.util import strtobool
import functools
import json
import os
import pickle

import cloudpickle
import numpy as np

from synmod import constants
from synmod import features as F
from synmod import models as M
from synmod.utils import get_logger, JSONEncoderPlus


def synthesize(**kwargs):
    """API to synthesize features, data and model"""
    strargs = []
    for key, value in kwargs.items():
        strargs.append(f"-{key}")
        strargs.append(f"{value}")
    return main(strargs=strargs)


def main(strargs=None):
    """Parse args and launch pipeline"""
    parser = argparse.ArgumentParser("python synmod")
    # Required arguments
    required = parser.add_argument_group("Required parameters")
    required.add_argument("-output_dir", help="Output directory", required=True)
    required.add_argument("-num_features", help="Number of features",
                          type=int, required=True)
    required.add_argument("-num_instances", help="Number of instances",
                          type=int, required=True)
    required.add_argument("-synthesis_type", help="Type of data/model synthesis to perform",
                          choices=[constants.TEMPORAL, constants.TABULAR], required=True)
    # Optional common arguments
    common = parser.add_argument_group("Common optional parameters")
    common.add_argument("-fraction_relevant_features", help="Fraction of features relevant to model",
                        type=float, default=1)
    common.add_argument("-num_interactions", help="number of pairwise in aggregation model (default 0)",
                        type=int, default=0)
    common.add_argument("-include_interaction_only_features", help="include interaction-only features in aggregation model"
                        " in addition to linear + interaction features (excluded by default)", type=strtobool)
    common.add_argument("-seed", help="Seed for RNG, random by default",
                        default=None, type=int)
    common.add_argument("-write_outputs", help="flag to enable writing outputs (alternative to using python API)",
                        type=strtobool)
    common.add_argument("-feature_type_distribution", help="option to specify distribution of binary/categorical/numeric"
                        "features types", nargs=3, type=float, default=[0.25, 0.25, 0.50])
    # Temporal synthesis arguments
    temporal = parser.add_argument_group("Temporal synthesis parameters")
    temporal.add_argument("-expected_sequence_length", help="Expected length of regularly sampled sequence",
                          type=int)
    # TODO: Make sequences dependent on windows by default to avoid unpredictability
    temporal.add_argument("-sequences_independent_of_windows", help="If enabled, Markov chain sequence data doesn't depend on timesteps being"
                          " inside vs. outside the window (default random)", type=strtobool, dest="window_independent")
    temporal.set_defaults(window_independent=None)
    temporal.add_argument("-model_type", help="type of model (classifier/regressor) - default random",
                          choices=[constants.CLASSIFIER, constants.REGRESSOR], default=constants.REGRESSOR)
    temporal.add_argument("-standardize_features", help="add feature standardization (0 mean, 1 SD) to model",
                          type=strtobool)
    temporal.add_argument("-observation_probability", help="The probability of observing a given feature value at any time point. Can be "
                                                           "either a single probability applied to all features (i.e. '0.1') or a "
                                                           "comma-seperated list of probabilities (i.e. '0.1,0.4,0.9) corresponding "
                                                           "to each feature. Default is 1 applied to all  features.",
                          type=str, default="1.0")
    temporal.add_argument("-feature_interactions", help="Number of feature interactions to include",
                          type=int)
    temporal.add_argument("-max_interactions_per_feature", help="Maximum number of other features any one feature can interact with",
                          type=int)
    temporal.add_argument("-feature_interaction_probability", help="The probability of one feature having an interaction with any other feature.",
                          type=float)
    temporal.add_argument("-interact_window_range",
                          help="Defines which time points before the current time point can be used for dependencies between features.",
                          type=str)
    temporal.add_argument("-interact_window_size",
                          help="The size of the interaction window (how many of the previous time points for feature A can influence the value of feature B)",
                          type=int)
    temporal.add_argument("-categorical_stability_scaler", help="Scaling factor for how much more likely a categorical variable is to keep its value over time point", type=float)

    args = parser.parse_args(args=strargs)
    if args.synthesis_type == constants.TEMPORAL:
        if args.expected_sequence_length is None:
            parser.error(f"-sequence_length required for -synthesis_type {constants.TEMPORAL}")
        elif args.expected_sequence_length <= 1:
            parser.error(f"-sequence_length must be greater than 1 for synthesis_type {constants.TEMPORAL}")
    else:
        args.expected_sequence_length = 1
    return pipeline(args)


def configure(args):
    """Configure arguments before execution"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.rng = np.random.default_rng(args.seed)
    args.logger = get_logger(__name__, f"{args.output_dir}/synmod.log")
    if args.window_independent is None:
        args.window_independent = args.rng.choice([True, False])


def pipeline(args):
    """Pipeline"""
    configure(args)
    args.logger.info(f"Begin generating sequence data with args: {args}")
    features = generate_features(args)
    instances = generate_instances(args, features)
    model = M.get_model(args, features, instances)
    ground_truth_estimation(args, features, instances, model)
    write_outputs(args, features, instances, model)
    return features, instances, model


def generate_features(args):
    """Generate features"""
    def check_feature_variance(args, feature):
        """Check variance of feature's raw/temporally aggregated values"""
        if args.synthesis_type == constants.TABULAR:
            instances = np.array([feature.sample() for _ in range(constants.VARIANCE_TEST_COUNT)])
            aggregated = instances
        else:
            instances = np.array([feature.sample(args.expected_sequence_length) for _ in range(constants.VARIANCE_TEST_COUNT)])
            left, right = feature.window
            aggregated = feature.aggregation_fn.operate(instances[:, left: right + 1])
        return np.var(aggregated) > 1e-10

    #Defining observation probabilities
    try:
        obs_prob = args.observation_probability.split(",")
        if len(obs_prob) == args.num_features:
            obs_prob = np.array([float(x) for x in obs_prob])
        elif len(obs_prob) == 1:
            obs_prob = np.array([float(args.observation_probability) for x in range(args.num_features)])
        else:
            raise Exception(f"Argument 'observation_probability' is not valid. Number of probabilities is {len(obs_prob)} and should be either 1 or {args.num_features}.")
    except:
        raise Exception(f"Argument 'observation_probability' is not numeric.")

    # TODO: allow across-feature interactions
    features = [None] * args.num_features
    fid = 0
    while fid < args.num_features:
        feature = F.get_feature(args, str(fid))
        feature.observation_probability = obs_prob[fid]

        possible_feature_to_depend_on = np.concatenate((np.random.choice([x for x in range(args.num_features) if x != fid], args.max_interactions_per_feature), [None]))
        probs = [args.feature_interaction_probability/(len(possible_feature_to_depend_on)-1) for x in possible_feature_to_depend_on]
        probs[-1] = 1-args.feature_interaction_probability
        f_depend_ids = [x for x in np.random.choice(possible_feature_to_depend_on, p=probs, size=args.max_interactions_per_feature) if x is not None]

        dependency_window = [int(x) for x in args.interact_window_range.split(",")]
        w_start = max(dependency_window)
        w_end = min(dependency_window)
        window_possible = list(range(-w_start, -(w_end+1)))

        dependencies = []
        for d in f_depend_ids:
            d_scale_factor = np.random.uniform(-1, 1)
            window_start = np.random.choice(window_possible, size=1)
            window_end = min(0, window_start + args.interact_window_size)
            dependencies.append((d, d_scale_factor, window_start, window_end, np.mean))
        feature.dependencies = dependencies

        if not check_feature_variance(args, feature):
            # Reject feature if its raw/aggregated values have low variance
            args.logger.info(f"Rejecting feature {feature.__class__} due to low variance")
            continue
        features[fid] = feature
        fid += 1
    return features


def sample_with_dependency(args, features, cur_seq_len, **kwargs):
    prev_time_feat_vals = np.zeros((len(features), cur_seq_len))


    for feature_id, feature in enumerate(features):
        cur_state = None
        for timepoint in range(cur_seq_len):

            mask = np.random.choice([0, 1], size=1, p=[1 - features[feature_id].observation_probability, features[feature_id].observation_probability])
            f_t_val, cur_state = feature.sample_single_MC_timepoint(args, cur_state, timepoint, prev_time_feat_vals, feature_id, feature.dependencies, **kwargs)
            prev_time_feat_vals[feature_id, timepoint] = f_t_val if mask.item() == 1 else np.nan

    return prev_time_feat_vals


def generate_instances(args, features):
    """Generate instances"""
    if args.synthesis_type == constants.TABULAR:
        instances = np.empty((args.num_instances, args.num_features))
        for sid in range(args.num_instances):
            instances[sid] = [feature.sample() for feature in features]
    else:
        seq_lengths = np.random.geometric(p=(1/args.expected_sequence_length), size=args.num_instances)
        max_len = np.max(seq_lengths).item()
        instances = []
        for instance_id in range(args.num_instances):
            cur_seq_len = seq_lengths[instance_id]

            if args.feature_interactions == 0:
                instance = [feature.sample(cur_seq_len) for feature in features]
            else:
                instance = sample_with_dependency(args, features, cur_seq_len)
            instance = np.pad(instance, pad_width=((0,0),(0,max_len-instance.shape[-1])), constant_values=np.nan)
            instances.append(instance)
    return np.stack(instances)


def generate_labels(model, instances):
    """Generate labels"""
    # TODO: decide how to handle multivariate case
    # TODO: joint generation of labels and features
    return model.predict(instances)


def ground_truth_estimation(args, features, instances, model):
    """Estimate and tag ground truth importance of features"""
    # pylint: disable = protected-access
    args.logger.info("Begin estimating ground truth effects")
    relevant_features = functools.reduce(set.union, model.relevant_feature_map, set())
    matrix = model._aggregator.operate(instances)
    zvec = np.zeros(args.num_features)
    for idx, feature in enumerate(features):
        if idx not in relevant_features:
            continue
        feature.important = True
        if args.model_type == constants.REGRESSOR:
            if args.num_interactions > 0:
                args.logger.info("Ground truth importance for interacting features not worked out")
                feature.effect_size = 1  # TODO: theory worked out only for non-interacting features
            else:
                # Compute effect size: 2 * covar(Y, g(X))
                fvec = np.copy(zvec)
                fvec[idx] = 1
                alpha = model._polynomial_fn(fvec, 1) - model._polynomial_fn(zvec, 1)  # Linear coefficient
                feature.effect_size = 2 * alpha**2 * np.var(matrix[:, idx])
        else:
            args.logger.info("Ground truth importance for classifier not well-defined")
            feature.effect_size = 1  # Ground truth importance score for classifier not well-defined
        if args.synthesis_type == constants.TEMPORAL:
            feature.window_important = True
            left, right = feature.window
            # TODO: Confirm these fields are correct when sequences have the same in- and out-distributions
            feature.window_ordering_important = feature.aggregation_fn.ordering_important
            feature.ordering_important = (right - left + 1 < args.expected_sequence_length) or feature.window_ordering_important
    args.logger.info("End estimating ground truth effects")


def write_outputs(args, features, instances, model):
    """Write outputs to file"""
    if not args.write_outputs:
        return
    with open(f"{args.output_dir}/{constants.FEATURES_FILENAME}", "wb") as features_file:
        cloudpickle.dump(features, features_file, protocol=pickle.DEFAULT_PROTOCOL)
    np.save(f"{args.output_dir}/{constants.INSTANCES_FILENAME}", instances)
    with open(f"{args.output_dir}/{constants.MODEL_FILENAME}", "wb") as model_file:
        cloudpickle.dump(model, model_file, protocol=pickle.DEFAULT_PROTOCOL)
    write_summary(args, features, model)


def write_summary(args, features, model):
    """Write summary of data generated"""
    config = dict(synthesis_type=args.synthesis_type,
                  num_instances=args.num_instances,
                  num_features=args.num_features,
                  sequence_length=args.expected_sequence_length,
                  model_type=model.__class__.__name__,
                  sequences_independent_of_windows=args.window_independent,
                  fraction_relevant_features=args.fraction_relevant_features,
                  num_interactions=args.num_interactions,
                  include_interaction_only_features=args.include_interaction_only_features,
                  seed=args.seed)
    # pylint: disable = protected-access
    features_summary = [feature.summary() for feature in features]
    model_summary = {}
    if args.synthesis_type == constants.TEMPORAL:
        model_summary["windows"] = [f"({window[0]}, {window[1]})" if window else None for window in model._aggregator._windows]
        model_summary["aggregation_fns"] = [agg_fn.__class__.__name__ for agg_fn in model._aggregator._aggregation_fns]
        model_summary["means"] = model._aggregator._means
        model_summary["stds"] = model._aggregator._stds
    model_summary["relevant_features"] = model.relevant_feature_names
    model_summary["polynomial"] = model.sym_polynomial_fn.__repr__()
    summary = dict(config=config, model=model_summary, features=features_summary)
    summary_filename = f"{args.output_dir}/{constants.SUMMARY_FILENAME}"
    args.logger.info(f"Writing summary to {summary_filename}")
    with open(summary_filename, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, cls=JSONEncoderPlus)
    return summary


if __name__ == "__main__":
    main()
