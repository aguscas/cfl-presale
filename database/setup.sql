CREATE DATABASE mydb;
GO
USE mydb;
GO
CREATE TABLE dbo.occupancy(
    OccupancyID INT,
    ParkingID VARCHAR(MAX),
    Occupancy INT,
    Heure DATE,
    Time TIME,
    Availability INT,
    Capacity INT,
    OccupancyRate FLOAT
);
