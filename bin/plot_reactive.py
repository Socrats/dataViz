import dataviz.distplot as distplot
import bin.paths as paths

# data_dict = {}
# for index, payoff_file in zip(paths.prefixes, paths.payoff_files):
#     actions_vector = np.load(payoff_file)
#     actions_vector = actions_vector[0][-1000:]
#     data_dict[''.join(['f', str(index)])] = actions_vector
#
# actions_df = pd.DataFrame(data_dict)
# actions_df = actions_df[['f1', 'f2', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f20']]
# # print(actions_df)
#
# actions_df.plot.hist(stacked=True, bins=paths.bins)
# actions_df.diff().hist(color='k', alpha=0.5, bins=paths.bins)
# plt.axhline(0, color='k')
# plt.show()

# actions_files = utils.get_files(paths.root_dir, "actions")
# data = pd.DataFrame(np.load(actions_files['f9'][1])[-1000:].transpose())
# bootstrap_plot(data, size=50, samples=500, color='grey')
# plt.show()


if __name__ == "__main__":
    sort_dict = "reactive"
    data = distplot.file2dataframe(paths.root_dir, "actions", 1000, sort_dict)
    # distplot.hist_by_tag(data, sort_dict, idx, paths.bins, paths.save_path, density=True)
    means, errors = distplot.calculate_avg_behavior(data)
    distplot.beautiful_box_plot(means, paths.save_path)
