-- FAOSTAT producers of a crop that devote at least 1% of their cropland to it in some year from 2018
-- to 2024. The producer sets in `jdluc/datasets/fdap.py` are these countries plus the countries with
-- thresholds. Area harvested (element 5312) and cropland (Land Use item 6620, element 5110, in
-- thousands of hectares) come from FAOSTAT's bulk files, and the ISO 3166 alpha-3 code for each M49
-- code from a public list, since the bulk files carry only M49 codes.
--
-- Run from the repo root, with the FAOSTAT item code: 661 for cocoa, 656 for coffee:
--   duckdb -cmd "SET VARIABLE item = '661';" -f tools/fdap-producers.sql
INSTALL httpfs;
LOAD httpfs;
INSTALL zipfs FROM community;
LOAD zipfs;

WITH crop AS (
    SELECT "Area Code (M49)" AS m49, CAST("Year" AS INTEGER) AS year, CAST("Value" AS DOUBLE) AS area_hectares
    FROM read_csv(
        'zip://https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip/Production_Crops_Livestock_E_All_Data_(Normalized).csv',
        all_varchar = true
    )
    WHERE "Item Code" = getvariable('item') AND "Element Code" = '5312' AND "Flag" != 'M' AND "Value" != ''
), cropland AS (
    SELECT "Area Code (M49)" AS m49, CAST("Year" AS INTEGER) AS year, CAST("Value" AS DOUBLE) * 1000 AS cropland_hectares
    FROM read_csv(
        'zip://https://bulks-faostat.fao.org/production/Inputs_LandUse_E_All_Data_(Normalized).zip/Inputs_LandUse_E_All_Data_(Normalized).csv',
        all_varchar = true
    )
    WHERE "Item Code" = '6620' AND "Element Code" = '5110' AND "Flag" != 'M' AND "Value" != ''
), iso AS (
    SELECT lpad("country-code", 3, '0') AS numeric, "alpha-3" AS iso_3166
    FROM read_csv('https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv', all_varchar = true)
)
SELECT iso_3166
FROM crop
JOIN cropland USING (m49, year)
JOIN iso ON iso.numeric = ltrim(crop.m49, '''')
WHERE year BETWEEN 2018 AND 2024 AND area_hectares > 0
GROUP BY iso_3166
HAVING MAX(area_hectares / cropland_hectares) >= 0.01
ORDER BY iso_3166;
