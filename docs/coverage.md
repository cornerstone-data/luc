# Coverage

## Countries

The pipeline produces emissions factors for the following countries and territories, identified by ISO 3166-1 alpha-3 code.

### Territorial extent

A country's footprint is its **full national boundary** (World Bank admin-0) intersected with the ten-degree tiles the source datasets actually publish. A country appears below when it touches at least one published tile *and* carries crop production for the pipeline to attribute emissions to.

The first half is a property of the source data rather than a curated list, so the set can shift when an upstream dataset is refreshed. The second half is a list: eight countries touch a published tile but grow nothing the pipeline can produce a factor for — Saint Barthélemy, Gibraltar, Greenland, Saint Martin, Monaco, Nauru, Tuvalu and the Holy See — and no property of the geometry separates them, so they are named explicitly as `worldbank_jurisdictions.UNPRODUCTIVE_ISO_3166S` and excluded there — by `get_all_iso_3166s`, which is what `--backfill` runs, and by `tools/build-tiled-countries.py`, which reads the same constant. The table below and `validation/data/tiled_iso_3166s.json` are the same 220 codes; regenerating that file is what keeps them so.

| Country                               | ISO 3166-1 alpha-3 |
| ------------------------------------- | ------------------ |
| Afghanistan                           | AFG                |
| Åland Islands                         | ALA                |
| Albania                               | ALB                |
| Algeria                               | DZA                |
| American Samoa                        | ASM                |
| Andorra                               | AND                |
| Angola                                | AGO                |
| Anguilla                              | AIA                |
| Antigua and Barbuda                   | ATG                |
| Argentina                             | ARG                |
| Armenia                               | ARM                |
| Aruba                                 | ABW                |
| Australia                             | AUS                |
| Austria                               | AUT                |
| Azerbaijan                            | AZE                |
| Bahamas                               | BHS                |
| Bahrain                               | BHR                |
| Bangladesh                            | BGD                |
| Barbados                              | BRB                |
| Belarus                               | BLR                |
| Belgium                               | BEL                |
| Belize                                | BLZ                |
| Benin                                 | BEN                |
| Bermuda                               | BMU                |
| Bhutan                                | BTN                |
| Bolivia                               | BOL                |
| Bonaire, Sint Eustatius and Saba      | BES                |
| Bosnia and Herzegovina                | BIH                |
| Botswana                              | BWA                |
| Brazil                                | BRA                |
| Brunei Darussalam                     | BRN                |
| Bulgaria                              | BGR                |
| Burkina Faso                          | BFA                |
| Burundi                               | BDI                |
| Cambodia                              | KHM                |
| Cameroon                              | CMR                |
| Canada                                | CAN                |
| Cayman Islands                        | CYM                |
| Central African Republic              | CAF                |
| Chad                                  | TCD                |
| Chile                                 | CHL                |
| China                                 | CHN                |
| Colombia                              | COL                |
| Comoros                               | COM                |
| Congo                                 | COG                |
| Congo, The Democratic Republic of the | COD                |
| Costa Rica                            | CRI                |
| Côte d'Ivoire                         | CIV                |
| Croatia                               | HRV                |
| Cuba                                  | CUB                |
| Curaçao                               | CUW                |
| Cyprus                                | CYP                |
| Czechia                               | CZE                |
| Denmark                               | DNK                |
| Djibouti                              | DJI                |
| Dominica                              | DMA                |
| Dominican Republic                    | DOM                |
| Ecuador                               | ECU                |
| Egypt                                 | EGY                |
| El Salvador                           | SLV                |
| Equatorial Guinea                     | GNQ                |
| Eritrea                               | ERI                |
| Estonia                               | EST                |
| Eswatini                              | SWZ                |
| Ethiopia                              | ETH                |
| Fiji                                  | FJI                |
| Finland                               | FIN                |
| France                                | FRA                |
| French Guiana                         | GUF                |
| French Southern Territories           | ATF                |
| Gabon                                 | GAB                |
| Gambia                                | GMB                |
| Georgia                               | GEO                |
| Germany                               | DEU                |
| Ghana                                 | GHA                |
| Greece                                | GRC                |
| Grenada                               | GRD                |
| Guadeloupe                            | GLP                |
| Guatemala                             | GTM                |
| Guernsey                              | GGY                |
| Guinea                                | GIN                |
| Guinea-Bissau                         | GNB                |
| Guyana                                | GUY                |
| Haiti                                 | HTI                |
| Honduras                              | HND                |
| Hong Kong                             | HKG                |
| Hungary                               | HUN                |
| Iceland                               | ISL                |
| India                                 | IND                |
| Indonesia                             | IDN                |
| Iran                                  | IRN                |
| Iraq                                  | IRQ                |
| Ireland                               | IRL                |
| Isle of Man                           | IMN                |
| Israel                                | ISR                |
| Italy                                 | ITA                |
| Jamaica                               | JAM                |
| Japan                                 | JPN                |
| Jersey                                | JEY                |
| Jordan                                | JOR                |
| Kazakhstan                            | KAZ                |
| Kenya                                 | KEN                |
| Kiribati                              | KIR                |
| Kosovo                                | XKX                |
| Kuwait                                | KWT                |
| Kyrgyzstan                            | KGZ                |
| Laos                                  | LAO                |
| Latvia                                | LVA                |
| Lebanon                               | LBN                |
| Lesotho                               | LSO                |
| Liberia                               | LBR                |
| Libya                                 | LBY                |
| Liechtenstein                         | LIE                |
| Lithuania                             | LTU                |
| Luxembourg                            | LUX                |
| Macao                                 | MAC                |
| Madagascar                            | MDG                |
| Malawi                                | MWI                |
| Malaysia                              | MYS                |
| Maldives                              | MDV                |
| Mali                                  | MLI                |
| Malta                                 | MLT                |
| Martinique                            | MTQ                |
| Mauritania                            | MRT                |
| Mauritius                             | MUS                |
| Mayotte                               | MYT                |
| Mexico                                | MEX                |
| Micronesia, Federated States of       | FSM                |
| Moldova                               | MDA                |
| Mongolia                              | MNG                |
| Montenegro                            | MNE                |
| Montserrat                            | MSR                |
| Morocco                               | MAR                |
| Mozambique                            | MOZ                |
| Myanmar                               | MMR                |
| Namibia                               | NAM                |
| Nepal                                 | NPL                |
| Netherlands                           | NLD                |
| New Caledonia                         | NCL                |
| New Zealand                           | NZL                |
| Nicaragua                             | NIC                |
| Niger                                 | NER                |
| Nigeria                               | NGA                |
| Norfolk Island                        | NFK                |
| North Korea                           | PRK                |
| North Macedonia                       | MKD                |
| Norway                                | NOR                |
| Oman                                  | OMN                |
| Pakistan                              | PAK                |
| Palau                                 | PLW                |
| Palestine, State of                   | PSE                |
| Panama                                | PAN                |
| Papua New Guinea                      | PNG                |
| Paraguay                              | PRY                |
| Peru                                  | PER                |
| Philippines                           | PHL                |
| Poland                                | POL                |
| Portugal                              | PRT                |
| Puerto Rico                           | PRI                |
| Qatar                                 | QAT                |
| Réunion                               | REU                |
| Romania                               | ROU                |
| Russian Federation                    | RUS                |
| Rwanda                                | RWA                |
| Saint Kitts and Nevis                 | KNA                |
| Saint Lucia                           | LCA                |
| Saint Pierre and Miquelon             | SPM                |
| Saint Vincent and the Grenadines      | VCT                |
| Samoa                                 | WSM                |
| San Marino                            | SMR                |
| Sao Tome and Principe                 | STP                |
| Saudi Arabia                          | SAU                |
| Senegal                               | SEN                |
| Serbia                                | SRB                |
| Seychelles                            | SYC                |
| Sierra Leone                          | SLE                |
| Singapore                             | SGP                |
| Sint Maarten (Dutch part)             | SXM                |
| Slovakia                              | SVK                |
| Slovenia                              | SVN                |
| Solomon Islands                       | SLB                |
| Somalia                               | SOM                |
| South Africa                          | ZAF                |
| South Korea                           | KOR                |
| South Sudan                           | SSD                |
| Spain                                 | ESP                |
| Sri Lanka                             | LKA                |
| Sudan                                 | SDN                |
| Suriname                              | SUR                |
| Svalbard and Jan Mayen                | SJM                |
| Sweden                                | SWE                |
| Switzerland                           | CHE                |
| Syria                                 | SYR                |
| Tajikistan                            | TJK                |
| Tanzania                              | TZA                |
| Thailand                              | THA                |
| Timor-Leste                           | TLS                |
| Togo                                  | TGO                |
| Tonga                                 | TON                |
| Trinidad and Tobago                   | TTO                |
| Tunisia                               | TUN                |
| Türkiye                               | TUR                |
| Turkmenistan                          | TKM                |
| Turks and Caicos Islands              | TCA                |
| Uganda                                | UGA                |
| Ukraine                               | UKR                |
| United Arab Emirates                  | ARE                |
| United Kingdom                        | GBR                |
| United States                         | USA                |
| Uruguay                               | URY                |
| Uzbekistan                            | UZB                |
| Vanuatu                               | VUT                |
| Venezuela                             | VEN                |
| Vietnam                               | VNM                |
| Virgin Islands, British               | VGB                |
| Virgin Islands, U.S.                  | VIR                |
| Wallis and Futuna                     | WLF                |
| Yemen                                 | YEM                |
| Zambia                                | ZMB                |
| Zimbabwe                              | ZWE                |

## Crops

### sLUC (statistical)

The statistical pipeline covers the following 32 crops, identified by IFPRI MapSPAM crop code.

| Crop           | MapSPAM code |
| -------------- | ------------ |
| ARABICA_COFFEE | ACOF         |
| BANANA         | BANA         |
| BARLEY         | BARL         |
| BEAN           | BEAN         |
| CASSAVA        | CASS         |
| CHICKPEA       | CHIC         |
| COCONUT        | CNUT         |
| COTTON         | COTT         |
| COWPEA         | COWP         |
| GROUNDNUT      | GROU         |
| LENTIL         | LENT         |
| MAIZE          | MAIZ         |
| OILPALM        | OILP         |
| OTHER_OILCROPS | OOIL         |
| OTHER_PULSES   | OPUL         |
| PEARL_MILLET   | PMIL         |
| PIGEONPEA      | PIGE         |
| PLANTAIN       | PLNT         |
| POTATO         | POTA         |
| RAPESEED       | RAPE         |
| RICE           | RICE         |
| ROBUSTA_COFFEE | RCOF         |
| SESAME_SEED    | SESA         |
| SMALL_MILLET   | SMIL         |
| SORGHUM        | SORG         |
| SOYBEAN        | SOYB         |
| SUGARBEET      | SUGB         |
| SUGARCANE      | SUGC         |
| SUNFLOWER      | SUNF         |
| SWEET_POTATO   | SWPO         |
| WHEAT          | WHEA         |
| YAM            | YAMS         |

### jdLUC (jurisdictional-direct)

The jurisdictional-direct pipeline (USA only) covers the following 11 crops, mapped to USDA NASS CDL crop classes.

| Crop      | USDA NASS CDL class(es)                 |
| --------- | --------------------------------------- |
| BARLEY    | BARLEY                                  |
| BEAN      | DRY_BEANS                               |
| COTTON    | COTTON                                  |
| MAIZE     | CORN                                    |
| POTATO    | POTATOES                                |
| RICE      | RICE                                    |
| SORGHUM   | SORGHUM                                 |
| SOYBEAN   | SOYBEANS                                |
| SUGARBEET | SUGARBEETS                              |
| SUGARCANE | SUGARCANE                               |
| WHEAT     | DURUM_WHEAT, SPRING_WHEAT, WINTER_WHEAT |
