from utils import *

plt.close("all")

# fig1, ax, _ = plot_one_manifold("Inchon_0810", env_id=1, reflect_z=True)
# ax.view_init(elev=20, azim=135)

# fig2, ax, _ = plot_two_manifolds("Inchon_0810", env_id=1, add_supervised_dimension=True)
# ax.view_init(elev=20, azim=175)

fig3, ax, _ = plot_one_manifold("Inchon_0812", env_id=1, reflect_y=False)
ax.view_init(elev=50, azim=105)

fig4, ax, _ = plot_two_manifolds("Inchon_0812", env_id=1, add_supervised_dimension=True)
ax.view_init(elev=20, azim=175)

fig3.savefig("Inchon_0812_single_map.pdf", dpi=500)
fig4.savefig("Inchon_0812_both_maps.pdf", dpi=500)

plt.show()
