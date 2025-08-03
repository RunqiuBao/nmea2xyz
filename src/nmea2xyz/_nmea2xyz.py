from pyproj import Transformer
import numpy
import matplotlib.pyplot as plt
import os.path


def rotation_from_vectors(vec1, vec2):
    """Compute rotation matrix that aligns vec1 to vec2"""
    vec1 = vec1 / numpy.linalg.norm(vec1)
    vec2 = vec2 / numpy.linalg.norm(vec2)
    v = numpy.cross(vec1, vec2)
    c = numpy.dot(vec1, vec2)
    s = numpy.linalg.norm(v)
    if s == 0:
        return numpy.eye(3)
    vx = numpy.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R_mat = numpy.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R_mat


def nmea_to_decimal(coord, direction):
    """Convert NMEA coordinate format to decimal degrees."""
    degrees = int(coord[:2]) if direction in ['N', 'S'] else int(coord[:3])
    minutes = float(coord[len(str(degrees)):])
    decimal = degrees + minutes / 60.0
    return -decimal if direction in ['S', 'W'] else decimal


def convert_nmea_to_xyz(
    nmea_file_path: str,
    indices_rotation_anchor_points: list,
    visz: bool
) -> None:
    """
    Convert NMEA data to XYZ coordinates.
    Args:
        nmea_file_path (str): Path to the input NMEA data file.
        indices_rotation_anchor_points (list): Indices of points to rotate trajectory to align.
        visz (bool): Whether to visualize the result trajectory in matplotlib.
    """
    # NMEA sentence
    # $GNGGA,091001.51,3542.8337549,N,13945.6313542,E,4,12,0.77,21.671,M,39.386,M,1.5,0000*5E
    with open(nmea_file_path, 'r') as file:
        raw_nmea_list = file.readlines()

    nmea_list = []
    for raw_nmea in raw_nmea_list:
        nmea_list.append(raw_nmea.split(','))
    lats, lons, alts = [], [], []
    transformer = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    for nmea in nmea_list:
        lat = nmea_to_decimal(nmea[2], nmea[3]) 
        lon = nmea_to_decimal(nmea[4], nmea[5])
        alt = float(nmea[9]) + float(nmea[11])
        lats.append(lat)
        lons.append(lon)
        alts.append(alt)
    x, y, z = transformer.transform(lons, lats, alts)

    traj = numpy.concatenate([numpy.array(x)[:, None], numpy.array(y)[:, None], numpy.array(z)[:, None]], axis=1)
    mean = traj.mean(axis=0)
    centered_traj = traj - mean

    # R_align = rotation_from_vectors(numpy.cross((centered_traj[500, :] - centered_traj[170, :]), (centered_traj[280, :] - centered_traj[170, :])), numpy.array([0, 0, 1]))
    R_align = rotation_from_vectors(
        numpy.cross((centered_traj[indices_rotation_anchor_points[2], :] - centered_traj[indices_rotation_anchor_points[0], :]), (centered_traj[indices_rotation_anchor_points[1], :] - centered_traj[indices_rotation_anchor_points[0], :])),
        numpy.array([0, 0, 1]))
    aligned_traj = (R_align @ centered_traj.T).T

    if visz:
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(*centered_traj.T)
        ax1.plot(*centered_traj.T[:, indices_rotation_anchor_points], marker='o', markersize=10)
        ax1.set_title("Original")
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(*aligned_traj.T)
        
        ax2.set_title("Rotation Aligned")
        plt.show()

    # output to tum format gt
    with open(os.path.join(os.path.dirame(nmea_file_path), os.path.basename(nmea_file_path).split('.')[0] + "_tumformat.txt"), 'w') as file:
        for ipoint in range(aligned_traj.shape[0]):
            line = str(ipoint) + " " + str(aligned_traj[ipoint, 0]) + str(aligned_traj[ipoint, 1]) + " " + str(aligned_traj[ipoint, 2]) + " 0 0 0 1\n"
            file.write(line)
