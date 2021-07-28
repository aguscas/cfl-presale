from pathlib import Path

COMET_PROJECT_NAME = "cfl-eta-phase1"
COMET_WORKSPACE = "gus0k"


GEN_FOLDER = Path("~/github/cfl-savvy-freight-eta/generated_data/").expanduser()
GEN_FOLDER.mkdir(parents=True, exist_ok=True)

DATA_FOLDER = Path("~/github/cfl-savvy-freight-eta/data/").expanduser()

RAWDATA_PATH = DATA_FOLDER / "excel2.csv"


MONGO_URL = "mongodb://devroot:devroot@localhost:27017"

ID_LEFT = "_id_left"
DEVICE_ID = "ITSS_TelematicsDeviceID"
APPLICATION_ID = "ITSS_TelematicsApplicationID"
TRANSPORT_ID = "ITSS_TransportDeviceID"
CREATEDAT = "createdAt"
ZIP = "zip"
CITY = "city"
COUNTRY_LEFT = "country_left"
ALTITUDE = "altitude"
LONGITUDE = "longitude"
SPEED_KMPH = "speed_kmph"
HEADING_DEG = "heading_deg"
DATE = "date"
LATITUDE = "latitude"
ACCURACY = "accuracy"
GEOMETRY = "geometry"
INDEX_RIGHT = "index_right"
ID_RIGHT = "_id_right"
POI_ID = "poi_Id"
POI_NAME = "poi_Name"
POI_REFERENCE = "poi_Reference"
CATEGORY = "category"
COUNTRY_RIGHT = "country_right"
NAME = "name"
TIMEZONE = "timezone"
SHAPE = "shape"
ASSOCIATEDCODE_UIC = "associatedCode_UIC"
IN_REFERENCE = "in_reference"
OUT_REFERENCE = "out_reference"
RADIUS = "radius"
CLUSTER = "cluster"
NEXT_CITY = "next_city"
TIME_BETWEEN_CITIES = "time_between_cities"

# Excel definition

ORDER_ID = "Order Number"
ORIGIN = "origin"
DESTINATION = "destination"
PLANNED_DEPARTURE_DATE = "planned_departure"
EFFECTIVE_DEPARTURE_DATE = "effective_departure"
PLANNED_ARRIVAL_DATE = "planned_arrival"
EFFECTIVE_ARRIVAL_DATE = "effective_arrival"
TARGET = "target"
STARTING_DATE = "starting_date"
RESOURCE_ID = "Resource Id"

PO_DATASET_PATH = GEN_FOLDER / "po_dataset.csv"
S2D = 3600 * 24
