"""Orthographic cluster and correlation maps (manuscript Figs. 2-9, Appendix A)."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .metrics import corr01_share

PALETTE = ["#f27a7a", "#a3c76d", "#6fb6d9", "#b8a8d9", "#f7d774"]


def _projection(lon_range, lat_range, central_lon, central_lat):
    if (lon_range[1] - lon_range[0] > 90) or (lat_range[1] - lat_range[0] > 90):
        return ccrs.Robinson(central_longitude=central_lon)
    return ccrs.Orthographic(central_longitude=central_lon, central_latitude=central_lat)


def _add_01corr_share_textbox(ax, share, title_fontsize):
    ax.text(0.98, 0.98, f"0.1corr share: {share:.0f}%", transform=ax.transAxes,
            fontsize=title_fontsize * 2 / 3, ha='right', va='top', zorder=20,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.85))


def map_clusters(lon, lat, labels, title, out_path, centroids=None,
                  central_lon=None, central_lat=None, lon_range=None, lat_range=None,
                  figsizeh=24, figsizev=26):
    """Orthographic map of cluster labels, saved to `out_path`."""
    lon, lat, labels = np.asarray(lon), np.asarray(lat), np.asarray(labels)
    central_lon = central_lon if central_lon is not None else (lon.max() + lon.min()) / 2
    central_lat = central_lat if central_lat is not None else (lat.max() + lat.min()) / 2
    lon_range = lon_range or [lon.min(), lon.max()]
    lat_range = lat_range or [lat.min(), lat.max()]
    projection = _projection(lon_range, lat_range, central_lon, central_lat)

    fig, ax = plt.subplots(figsize=(figsizeh, figsizev), subplot_kw={'projection': projection})
    ax.set_title(title, fontsize=36)

    scatter_handles = []
    for idx, label in enumerate(np.unique(labels)):
        mask = labels == label
        ax.scatter(lon[mask], lat[mask], marker='o', s=150, color=PALETTE[idx % len(PALETTE)],
                   transform=ccrs.PlateCarree())
        scatter_handles.append(mlines.Line2D([], [], marker='o', color=PALETTE[idx % len(PALETTE)],
                                              linestyle='None', markersize=24, label=f'Cluster {int(label)}'))

    if centroids is not None:
        centroids = np.asarray(centroids)
        ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', lw=2, s=100, color='black',
                   transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=2, edgecolor='black')
    if not ((lon_range[1] - lon_range[0] > 90) or (lat_range[1] - lat_range[0] > 90)):
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    plt.legend(handles=scatter_handles, fontsize=30, loc='upper center', bbox_to_anchor=(0.5, -0),
               ncol=len(scatter_handles), frameon=False)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def map_correlation(corr_df, cl, title, out_path, central_lon=None, central_lat=None,
                     lon_range=None, lat_range=None, figsizeh=24, figsizev=26):
    """
    Orthographic correlation map for one cluster: gridpoint correlation to
    the centroid, the medoid-coherence core (black dots), the single
    highest-correlation point (star), and the area-weighted "0.1corr share"
    textbox.
    """
    value_col, medoid_col = f"cluster_{cl}", f"medoid_corr_{cl}"
    lon, lat = corr_df.nodes_lon.values, corr_df.nodes_lat.values
    values = corr_df[value_col].values
    med_values = corr_df[medoid_col].values
    in_cluster = (corr_df["cluster"].values == cl)

    central_lon = central_lon if central_lon is not None else (lon.max() + lon.min()) / 2
    central_lat = central_lat if central_lat is not None else (lat.max() + lat.min()) / 2
    lon_range = lon_range or [lon.min(), lon.max()]
    lat_range = lat_range or [lat.min(), lat.max()]
    projection = _projection(lon_range, lat_range, central_lon, central_lat)

    idx_max = np.argmax(values)
    base_color = mcolors.to_rgb(PALETTE[cl % len(PALETTE)])
    dark_color = tuple(np.clip(np.array(base_color) * 0.6, 0, 1))
    cmap = mcolors.ListedColormap(mcolors.LinearSegmentedColormap.from_list(
        "custom_corr", [(0.0, "black"), (0.5, "white"), (0.75, base_color), (1.0, dark_color)]
    )(np.linspace(0, 1, 20)))
    bounds = np.linspace(-1, 1, 21)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(figsizeh, figsizev), subplot_kw={'projection': projection})
    sc = ax.scatter(lon, lat, c=values, cmap=cmap, norm=norm, marker='o', s=50,
                     edgecolors='none', transform=ccrs.PlateCarree())
    ax.scatter(lon[med_values > 0.1], lat[med_values > 0.1], color='black', s=1, marker='o',
               transform=ccrs.PlateCarree(), zorder=4)
    ax.scatter(lon[idx_max], lat[idx_max], color='black', s=200, marker='*', edgecolors='black',
               transform=ccrs.PlateCarree(), zorder=5)

    title_fontsize = 36
    share = corr01_share(values, med_values, lat, in_cluster)
    _add_01corr_share_textbox(ax, share, title_fontsize)

    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, boundaries=bounds, ticks=np.linspace(-1, 1, 11))
    cbar.set_label("Correlation", fontsize=26)
    plt.title(f"{title} {cl}", fontsize=title_fontsize)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def map_scalar_field(lon, lat, values, title, out_path, legend_label='', palette='Reds',
                      central_lon=None, central_lat=None, lon_range=None, lat_range=None,
                      figsizeh=10, figsizev=10):
    """Generic orthographic scalar-field map (e.g. a per-gridpoint diagnostic value)."""
    lon, lat = np.asarray(lon), np.asarray(lat)
    central_lon = central_lon if central_lon is not None else (lon.max() + lon.min()) / 2
    central_lat = central_lat if central_lat is not None else (lat.max() + lat.min()) / 2
    lon_range = lon_range or [lon.min(), lon.max()]
    lat_range = lat_range or [lat.min(), lat.max()]
    projection = _projection(lon_range, lat_range, central_lon, central_lat)

    fig = plt.figure(figsize=(figsizeh, figsizev))
    ax = plt.axes(projection=projection)
    ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='gray')
    if not ((lon_range[1] - lon_range[0] > 90) or (lat_range[1] - lat_range[0] > 90)):
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    sc = ax.scatter(lon, lat, c=values, cmap=palette, s=15, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(sc, orientation='vertical', shrink=0.6)
    cbar.set_label(legend_label)
    plt.title(title)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
