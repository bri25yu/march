from tqdm.auto import tqdm

from pandas import DataFrame

from march.experiments import available_experiments


ABS_DIFF_PERCENT_THRESHOLD = 2.0

available_experiments = available_experiments.copy()

baseline_experiment_cls = available_experiments.pop("BaselineExperiment")
baseline_num_params = baseline_experiment_cls().get_model().count_parameters()

df = []
for experiment_name, experiment_cls in tqdm(available_experiments.items(), desc="Checking experiment model param counts"):
    try:
        experiment_num_params = experiment_cls().get_model().count_parameters()
        df.append({"Experiment name": experiment_name, "Num params": experiment_num_params})
    except:
        print(f"Failed to get parameters for {experiment_name}")

df = DataFrame.from_dict(df)
df["Percent diff"] = 100 * (df["Num params"] - baseline_num_params) / baseline_num_params

diff_too_big_df = df[df["Percent diff"].abs() >= ABS_DIFF_PERCENT_THRESHOLD]

diff_too_big_df_for_display = diff_too_big_df.copy()
diff_too_big_df_for_display["Experiment name"] = diff_too_big_df_for_display["Experiment name"].apply(lambda v: v.removesuffix("Experiment"))
diff_too_big_df_for_display["Percent diff"] = diff_too_big_df_for_display["Percent diff"].apply(lambda v: f"{v:+.2f}%")
diff_too_big_df_for_display["Num params"] = diff_too_big_df_for_display["Num params"].apply(lambda v: f"{v:,}")

print(f"Experiments with params > +-{ABS_DIFF_PERCENT_THRESHOLD}% from the baseline:\n\n{diff_too_big_df_for_display.to_string(index=False)}")
