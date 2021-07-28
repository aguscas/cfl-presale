loaddata:
	docker cp data/last_position.json cfl-savvy-freight-eta_mongodb_1:/tmp/last_position.json
	docker exec cfl-savvy-freight-eta_mongodb_1 mongoimport -u "devroot" -p "devroot" --file /tmp/last_position.json --jsonArray
	docker cp data/poi_coordinates.json cfl-savvy-freight-eta_mongodb_1:/tmp/poi_coordinates.json
	docker exec cfl-savvy-freight-eta_mongodb_1 mongoimport -u "devroot" -p "devroot" --file /tmp/poi_coordinates.json --jsonArray

