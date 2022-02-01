from utils import *

fig1, _, _ = plot_one_manifold("Pisa_0430_1", add_supervised_dimension=True)

fig2, _, _ = plot_one_manifold("Portland_1005_2", reflect_y=True)

fig3, _, _ = plot_two_manifolds("Pisa_0430_1")

fig4, _, _ = plot_two_manifolds("Portland_1005_2")

fig5, _, _ = plot_two_manifolds("Pisa_0430_1", shuffle=True)

fig6, _, _ = plot_two_manifolds("Portland_1005_2", shuffle=True)

fig1.savefig("../manifold_figure/cue_poor_1map.pdf")
fig2.savefig("../manifold_figure/cue_rich_1map.pdf")
fig3.savefig("../manifold_figure/cue_poor_2map.pdf")
fig4.savefig("../manifold_figure/cue_rich_2map.pdf")
fig5.savefig("../manifold_figure/cue_poor_shuffle.pdf")
fig6.savefig("../manifold_figure/cue_rich_shuffle.pdf")
