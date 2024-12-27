# Sun position is calculated in ECI Frame
# Earth's Magnetic Field is calculated in ECEF Frame
# Satellite's Position is calculated in the ECEF Frame

# TRIAD ALGORITHMM
# GIVENS:
# Sun Vector + Magnetic Field Vector in the ECI Frame
# Sun Vector + Magnetic Field Vector in the Local Frame

# OUTPUTS:
# Rotation Matrix that describes the local frame wrt to ECI.

import numpy as np
from scipy.spatial.transform import Rotation as R
import pymap3d as pm
import wmm2020
from pyproj import Transformer
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import (
    GCRS,
    ITRS,
    get_body,
    get_sun,
    CartesianRepresentation
)
# Position of satellite, arbitrary for now
x_ecef = -6345.24
y_ecef = 1234.56
z_ecef = 4523.87
time_of_interest = Time.now()

## GIVENS
# Local Frame (Arbitrary for now, given by sensors)
# Sun Vector
sun_vector_local = [0.5, -0.7, 0.5]
# Magnetic Field Vector
mfield_vector_local = [-0.6, 0.3, 0.7]

# ECI Frame
# Sun Vector, GCRS = ECI
sun_gcrs = get_body("sun", time_of_interest, ephemeris="builtin")
sun_pos_eci = sun_gcrs.cartesian.xyz.to(u.km)   # Vector From Earth to Sun in ECI
print(sun_pos_eci)
sat_pos_ecef = CartesianRepresentation(x_ecef * u.km, y_ecef * u.km, z_ecef * u.km)
sat_pos_ecef = ITRS(sat_pos_ecef, obstime=time_of_interest)
sat_pos_eci = sat_pos_ecef.transform_to(GCRS(obstime=time_of_interest)).cartesian.xyz.to(u.km)  # Vector from Sat to Earth in ECI
print(sat_pos_eci)
sun_vector_eci = sun_pos_eci-sat_pos_eci    # Vector from Sat to Sun in ECI
print(sun_vector_eci)

# Magnetic Field Vector
# Define inputs (Arbitrary for now, geodetic coordinates)
# position
transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326")  # ECEF to WGS84
lat, lon, alt = transformer.transform(x_ecef, y_ecef, z_ecef)
date = time_of_interest.decimalyear  # date in decimal year
mag_field = wmm2020.wmm(lat, lon, alt, date)

B_north = mag_field["north"].values
B_east = mag_field["east"].values
B_down = mag_field["down"].values

# NED to ECEF
mfield_vector_ecef = pm.ned2ecef(B_north, B_east, B_down, lat, lon, alt)
# converting to correct form
mfield_vector_ecef = np.array([
    mfield_vector_ecef[0][0][0],
    mfield_vector_ecef[1][0][0],
    mfield_vector_ecef[2][0][0]
])

print("mfield vector in ECEF (nT):", mfield_vector_ecef)

# ECI
mfield_vector_itrs = ITRS(
    x=mfield_vector_ecef[0] * u.nT,
    y=mfield_vector_ecef[1] * u.nT,
    z=mfield_vector_ecef[2] * u.nT,
    representation_type=CartesianRepresentation,
    obstime=time_of_interest
)
mfield_vector_eci = mfield_vector_itrs.transform_to(GCRS(obstime=time_of_interest)).cartesian.xyz.to(u.nT)
print("mfield vector in ECI (nT):", mfield_vector_eci)

# triad
S_B = np.array(sun_vector_local)
S_N = np.array(sun_vector_eci)
m_B = np.array(mfield_vector_local)
m_N = np.array(mfield_vector_eci)

# normalize
S_B = S_B / np.linalg.norm(S_B)
S_N = S_N / np.linalg.norm(S_N)
m_B = m_B / np.linalg.norm(m_B)
m_N = m_N / np.linalg.norm(m_N)

# B-frame
t_1B = S_B
t_2B = np.cross(S_B, m_B)/np.linalg.norm(np.cross(S_B, m_B))
t_3B = np.cross(t_1B, t_2B)

# N-frame
t_1N = np.array(sun_vector_eci)
t_2N = np.cross(S_N, m_N)/np.linalg.norm(np.cross(S_N, m_N))
t_3N = np.cross(t_1N, t_2N)

Bt = np.column_stack((t_1B, t_2B, t_3B))
Nt = np.column_stack((t_1N, t_2N, t_3N))

rot_matrix = np.dot(Bt, Nt.T)
print(rot_matrix)