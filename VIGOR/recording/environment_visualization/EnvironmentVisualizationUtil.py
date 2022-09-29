
def get_plot_title(iteration: int, prefix: str = None) -> str:
    title = "Iteration: {:04d}".format(iteration)
    if prefix is not None:
        title = prefix+" "+title
    return title
