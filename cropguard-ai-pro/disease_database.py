"""
CropGuard AI Pro - Disease Database
Comprehensive crop disease reference with symptoms & treatments
"""

CROP_DISEASE_DB = {
    "tomato": {
        "name": "Tomato",
        "icon": "🍅",
        "diseases": {
            "Septoria Leaf Spot": {
                "severity": "Medium",
                "description": "Fungal disease causing small circular spots with dark borders and light gray centers, usually starting on lower leaves.",
                "symptoms": [
                    "Small circular spots with dark brown margins",
                    "Gray/tan centers with tiny black dots (fungal fruiting bodies)",
                    "Starts on older, lower leaves and spreads upward",
                    "Severe cases cause leaf yellowing and drop"
                ],
                "treatments": {
                    "chemical": [
                        "Chlorothalonil 75% WP @ 2g/L",
                        "Mancozeb 75% WP @ 2.5g/L",
                        "Copper oxychloride 50% WP @ 3g/L"
                    ],
                    "organic": [
                        "Neem oil spray every 7-10 days",
                        "Copper-based organic fungicide",
                        "Remove and destroy infected lower leaves"
                    ],
                    "prevention": [
                        "Avoid overhead watering, water at soil level",
                        "Space plants for good air circulation",
                        "Mulch to prevent soil splash onto leaves",
                        "Rotate crops, avoid planting tomato after tomato"
                    ]
                }
            },
            "Leaf Mold": {
                "severity": "Medium",
                "description": "Fungal disease common in humid, poorly ventilated conditions, causing pale yellow spots on upper leaf surface with olive-green mold underneath.",
                "symptoms": [
                    "Pale yellow spots on upper leaf surface",
                    "Olive-green to grayish-purple velvety mold on leaf underside",
                    "Leaves curl, wither, and drop in severe cases"
                ],
                "treatments": {
                    "chemical": [
                        "Chlorothalonil 75% WP @ 2g/L",
                        "Difenoconazole 25% EC @ 1ml/L"
                    ],
                    "organic": [
                        "Improve greenhouse/field ventilation",
                        "Neem oil spray",
                        "Baking soda spray solution"
                    ],
                    "prevention": [
                        "Reduce humidity around plants",
                        "Avoid overhead irrigation",
                        "Increase plant spacing for airflow"
                    ]
                }
            },
            "Mosaic Virus": {
                "severity": "High",
                "description": "Viral disease causing mottled light and dark green patterns on leaves, stunting plant growth. No chemical cure - management is preventive.",
                "symptoms": [
                    "Mottled yellow-green mosaic pattern on leaves",
                    "Curled, distorted, or narrow leaves",
                    "Stunted plant growth",
                    "Reduced fruit yield and quality"
                ],
                "treatments": {
                    "chemical": [
                        "No direct chemical cure - control aphid/insect vectors with Imidacloprid 17.8% SL @ 0.5ml/L"
                    ],
                    "organic": [
                        "Remove and destroy infected plants immediately",
                        "Control aphids with neem oil or insecticidal soap",
                        "Use reflective mulch to deter insect vectors"
                    ],
                    "prevention": [
                        "Use virus-resistant/tolerant varieties",
                        "Disinfect tools between plants",
                        "Control weeds that host the virus",
                        "Avoid handling healthy plants after infected ones"
                    ]
                }
            },
            "Yellow Leaf Curl Virus": {
                "severity": "High",
                "description": "Whitefly-transmitted viral disease causing upward leaf curling, yellowing, and severe stunting.",
                "symptoms": [
                    "Upward curling of leaflets",
                    "Yellowing of leaf margins and interveinal areas",
                    "Severe stunting of plant growth",
                    "Flower drop and reduced fruit set"
                ],
                "treatments": {
                    "chemical": [
                        "Control whitefly vector with Imidacloprid 17.8% SL @ 0.5ml/L",
                        "Thiamethoxam 25% WG @ 0.3g/L"
                    ],
                    "organic": [
                        "Yellow sticky traps for whitefly monitoring/control",
                        "Neem oil spray to deter whiteflies",
                        "Remove and destroy infected plants"
                    ],
                    "prevention": [
                        "Use virus-resistant varieties",
                        "Use insect-proof nets/screens in nurseries",
                        "Control whitefly populations early in season"
                    ]
                }
            },
            "Spider Mites": {
                "severity": "Medium",
                "description": "Tiny pest infestation causing stippled, bronzed leaves with fine webbing, worse in hot, dry conditions.",
                "symptoms": [
                    "Fine yellow/white stippling on leaves",
                    "Bronze or bronze-yellow discoloration",
                    "Fine silk webbing on leaf undersides",
                    "Leaf drop in severe infestations"
                ],
                "treatments": {
                    "chemical": [
                        "Abamectin 1.8% EC @ 0.5ml/L",
                        "Spiromesifen 22.9% SC @ 1ml/L"
                    ],
                    "organic": [
                        "Neem oil spray, focus on leaf undersides",
                        "Insecticidal soap spray",
                        "Introduce predatory mites as biological control"
                    ],
                    "prevention": [
                        "Maintain adequate humidity (mites thrive in dry conditions)",
                        "Regularly hose down plants to disrupt mite colonies",
                        "Avoid excess nitrogen fertilization"
                    ]
                }
            },
            "Late Blight": {
                "severity": "High",
                "description": "Water-soaked lesions turning brown/black with white fungal growth on underside in humid conditions.",
                "symptoms": [
                    "Dark water-soaked spots on leaves",
                    "White mold on leaf underside",
                    "Stem lesions with dark coloration",
                    "Fruit rot starting from calyx end"
                ],
                "treatments": {
                    "chemical": [
                        "Mancozeb 75% WP @ 2g/L water",
                        "Metalaxyl 8% + Mancozeb 64% WP @ 2g/L",
                        "Chlorothalonil 75% WP @ 2g/L",
                        "Propamocarb HCl 70% WP @ 1.5g/L"
                    ],
                    "organic": [
                        "Baking soda spray (1 tbsp/L water + few drops soap)",
                        "Copper fungicide (organic approved)",
                        "Neem oil 3% @ 5ml/L water",
                        "Compost tea foliar spray weekly"
                    ],
                    "prevention": [
                        "Remove infected plants immediately",
                        "Avoid overhead irrigation - use drip instead",
                        "Ensure good air circulation (60x45cm spacing)",
                        "Plant resistant varieties (Arka Rakshak, Arka Saurabh)",
                        "Apply preventive fungicide before monsoon"
                    ]
                }
            },
            "Early Blight": {
                "severity": "Medium",
                "description": "Concentric rings (target spots) on older leaves. Yellowing and defoliation from bottom up.",
                "symptoms": [
                    "Target-like concentric rings on leaves",
                    "Yellowing starting from lower leaves",
                    "Stem cankers near soil line",
                    "Fruit lesions near stem attachment"
                ],
                "treatments": {
                    "chemical": [
                        "Mancozeb 75% WP @ 2g/L",
                        "Azoxystrobin 23% SC @ 1ml/L",
                        "Difenoconazole 25% EC @ 1ml/L",
                        "Copper oxychloride 50% WP @ 3g/L"
                    ],
                    "organic": [
                        "Compost tea spray weekly",
                        "Baking soda + liquid soap solution",
                        "Neem oil @ 5ml/L with sticker",
                        "Garlic-chili extract spray"
                    ],
                    "prevention": [
                        "Rotate crops (3-4 year gap)",
                        "Remove and destroy crop debris",
                        "Mulch to prevent soil splash",
                        "Stake plants for airflow",
                        "Avoid working in wet fields"
                    ]
                }
            },
            "Bacterial Spot": {
                "severity": "Medium",
                "description": "Small water-soaked spots turning dark brown/black with yellow halos. Spreads rapidly in wet weather.",
                "symptoms": [
                    "Small dark spots with yellow halos",
                    "Leaf yellowing and curling",
                    "Raised scabby spots on fruit",
                    "Stem spots with bacterial ooze"
                ],
                "treatments": {
                    "chemical": [
                        "Streptomycin sulfate 90% @ 0.5g/L",
                        "Copper oxychloride 50% WP @ 3g/L",
                        "Kasugamycin 3% SL @ 2ml/L",
                        "Oxytetracycline 19.3% @ 1g/L"
                    ],
                    "organic": [
                        "Copper soap spray (organic grade)",
                        "Hydrogen peroxide 3% solution",
                        "Compost tea with beneficial bacteria",
                        "Baking soda + neem oil combination"
                    ],
                    "prevention": [
                        "Use certified disease-free seeds",
                        "Avoid working in wet fields",
                        "Sanitize tools with 10% bleach",
                        "Plant resistant varieties",
                        "Copper spray at 7-day intervals"
                    ]
                }
            },
            "Leaf Curl Virus": {
                "severity": "High",
                "description": "Leaves curl upward, become leathery and yellow. Stunted growth. Spread by whiteflies (Bemisia tabaci).",
                "symptoms": [
                    "Upward curling of leaves",
                    "Yellowing and thickening of veins",
                    "Stunted plant growth",
                    "Reduced fruit set and size"
                ],
                "treatments": {
                    "chemical": [
                        "Imidacloprid 17.8% SL @ 0.5ml/L (whitefly control)",
                        "Thiamethoxam 25% WG @ 0.3g/L",
                        "Buprofezin 25% SC @ 1ml/L",
                        "NOTE: No cure for virus - remove infected plants"
                    ],
                    "organic": [
                        "Neem oil @ 5ml/L for whitefly",
                        "Yellow sticky traps @ 10/acre",
                        "Reflective mulch (silver) to repel whiteflies",
                        "Release Encarsia formosa parasitoids"
                    ],
                    "prevention": [
                        "Use virus-resistant varieties (Arka Rakshak)",
                        "Control whitefly vectors from seedling stage",
                        "Remove infected plants immediately and burn",
                        "Avoid planting near infected fields",
                        "Use 40-mesh insect-proof nets"
                    ]
                }
            },
            "Healthy": {
                "severity": "Healthy",
                "description": "Plant appears healthy with normal green foliage and no visible disease symptoms.",
                "symptoms": [
                    "Uniform dark green color",
                    "No spots, lesions, or discoloration",
                    "Normal growth vigor and height",
                    "Healthy flower and fruit set"
                ],
                "treatments": {
                    "chemical": [
                        "No treatment needed",
                        "Preventive fungicide if high disease pressure forecast"
                    ],
                    "organic": [
                        "Continue organic practices",
                        "Compost tea foliar feed monthly",
                        "Seaweed extract @ 3ml/L every 15 days",
                        "Vermiwash spray for micronutrients"
                    ],
                    "prevention": [
                        "Maintain balanced nutrition (NPK + micronutrients)",
                        "Regular scouting every 3-4 days",
                        "Proper spacing (60x45 cm)",
                        "Crop rotation annually with legumes",
                        "Maintain field sanitation"
                    ]
                }
            }
        }
    },
    "potato": {
        "name": "Potato",
        "icon": "🥔",
        "diseases": {
            "Late Blight": {
                "severity": "High",
                "description": "Dark water-soaked lesions on leaves, rapid defoliation. White fungal growth on leaf underside in humid conditions.",
                "symptoms": [
                    "Dark water-soaked lesions on leaves",
                    "White cottony mold on leaf underside",
                    "Rapid defoliation within 3-5 days",
                    "Tuber rot in storage (reddish-brown dry rot)"
                ],
                "treatments": {
                    "chemical": [
                        "Mancozeb 75% WP @ 2.5g/L",
                        "Metalaxyl 8% + Mancozeb 64% WP @ 2g/L",
                        "Propamocarb HCl 70% WP @ 1.5g/L",
                        "Dimethomorph 50% WP @ 2g/L"
                    ],
                    "organic": [
                        "Copper sulfate spray (Bordeaux mixture 1%)",
                        "Baking soda solution (2g/L)",
                        "Compost tea with Trichoderma",
                        "Milk spray (1:9 dilution)"
                    ],
                    "prevention": [
                        "Destroy volunteer potatoes immediately",
                        "Hill soil around stems at 30 days",
                        "Harvest before frost and dry tubers",
                        "Store at 4°C with good ventilation",
                        "Use certified seed tubers"
                    ]
                }
            },
            "Early Blight": {
                "severity": "Medium",
                "description": "Dark brown concentric rings on older leaves. Defoliation starts from bottom. Tuber infection shows dry brown rot.",
                "symptoms": [
                    "Target spots with concentric rings",
                    "Lower leaf yellowing and browning",
                    "Stem lesions near soil line",
                    "Dry brown tuber rot under skin"
                ],
                "treatments": {
                    "chemical": [
                        "Mancozeb 75% WP @ 2g/L",
                        "Chlorothalonil 75% WP @ 2g/L",
                        "Azoxystrobin 23% SC @ 1ml/L",
                        "Pyraclostrobin 20% WG @ 1g/L"
                    ],
                    "organic": [
                        "Neem oil @ 5ml/L with sticker",
                        "Compost tea spray every 10 days",
                        "Baking soda + horticultural oil",
                        "Seaweed extract foliar feed"
                    ],
                    "prevention": [
                        "Rotate with non-solanaceous crops (cereals, legumes)",
                        "Remove and destroy infected plant debris",
                        "Adequate spacing (60x20 cm)",
                        "Balanced fertilization (avoid excess N)",
                        "Hilling at 25-30 days after planting"
                    ]
                }
            },
            "Black Scurf": {
                "severity": "Medium",
                "description": "Black sclerotia (soot-like spots) on tuber surface. Affects appearance and market value. Soil-borne Rhizoctonia solani.",
                "symptoms": [
                    "Black irregular spots on tuber skin",
                    "Rough skin texture",
                    "Reduced market grade",
                    "Stunted sprouts (pre-emergence damping off)"
                ],
                "treatments": {
                    "chemical": [
                        "Carbendazim 50% WP @ 1g/L seed treatment",
                        "Mancozeb 75% WP soil drench at planting",
                        "Thiophanate-methyl 70% WP @ 1g/L",
                        "Validamycin 3% L @ 2ml/L"
                    ],
                    "organic": [
                        "Hot water treatment (50°C for 10 min) before storage",
                        "Compost tea soil drench",
                        "Biofungicide Trichoderma harzianum @ 10g/kg seed",
                        "Mustard cake soil application @ 500kg/ha"
                    ],
                    "prevention": [
                        "Use certified seed tubers",
                        "Crop rotation (3+ years with cereals)",
                        "Soil solarization in summer",
                        "Avoid planting in infected soil",
                        "Proper storage at 10-12°C, 90% RH"
                    ]
                }
            },
            "Healthy": {
                "severity": "Healthy",
                "description": "Healthy potato plant with dark green compound leaves and normal tuber development.",
                "symptoms": [
                    "Dark green compound leaves",
                    "No spots, lesions, or yellowing",
                    "Normal upright stem growth",
                    "Healthy tubers with smooth skin"
                ],
                "treatments": {
                    "chemical": [
                        "Preventive mancozeb spray if late blight risk",
                        "Seed treatment with mancozeb @ 2.5g/kg"
                    ],
                    "organic": [
                        "Seaweed extract foliar feed at tuber initiation",
                        "Compost tea drench at flowering",
                        "Vermicompost @ 5 tons/ha"
                    ],
                    "prevention": [
                        "Certified seed potatoes from authorized source",
                        "Crop rotation (3 years minimum)",
                        "Proper hilling at 25-30 days",
                        "Adequate drainage - avoid waterlogging",
                        "Rogue out diseased plants immediately"
                    ]
                }
            }
        }
    },
    "corn": {
        "name": "Corn (Maize)",
        "icon": "🌽",
        "diseases": {
            "Northern Leaf Blight": {
                "severity": "High",
                "description": "Long cigar-shaped gray-green to tan lesions on leaves. Severe infection causes complete defoliation and yield loss.",
                "symptoms": [
                    "Long elliptical lesions (2-15 cm)",
                    "Gray-green to tan colored spots",
                    "Lesions merge in severe cases",
                    "Defoliation from top leaves downward"
                ],
                "treatments": {
                    "chemical": [
                        "Azoxystrobin 23% SC @ 1ml/L",
                        "Propiconazole 25% EC @ 1ml/L",
                        "Mancozeb 75% WP @ 2g/L",
                        "Tebuconazole 25% EC @ 1ml/L"
                    ],
                    "organic": [
                        "Neem oil @ 5ml/L with sticker",
                        "Compost tea foliar spray",
                        "Baking soda spray (preventive)",
                        "Cow urine dilution (1:10) spray"
                    ],
                    "prevention": [
                        "Plant resistant hybrids (e.g., Pusa Composite)",
                        "Rotate crops (2 years with legumes)",
                        "Remove and destroy crop residue",
                        "Avoid dense planting (60x20 cm)",
                        "Balanced NPK (120:60:40 kg/ha)"
                    ]
                }
            },
            "Common Rust": {
                "severity": "Medium",
                "description": "Small reddish-brown pustules on both leaf surfaces. Pustules rupture releasing rust-colored spores.",
                "symptoms": [
                    "Reddish-brown pustules on leaves",
                    "Powdery spore masses (rust color)",
                    "Yellowing around pustules",
                    "Reduced ear size and grain fill"
                ],
                "treatments": {
                    "chemical": [
                        "Propiconazole 25% EC @ 1ml/L",
                        "Tebuconazole 25% EC @ 1ml/L",
                        "Mancozeb 75% WP @ 2g/L",
                        "Difenoconazole 25% EC @ 1ml/L"
                    ],
                    "organic": [
                        "Neem oil spray @ 5ml/L",
                        "Compost tea foliar application",
                        "Milk spray (1:9 with water)",
                        "Sulfur dust (wettable sulfur 80% WP @ 3g/L)"
                    ],
                    "prevention": [
                        "Plant resistant varieties (e.g., Ganga-5)",
                        "Early planting to avoid peak rust season",
                        "Adequate spacing (60x20 cm)",
                        "Balanced nitrogen application",
                        "Remove volunteer corn plants"
                    ]
                }
            },
            "Stem Borer": {
                "severity": "High",
                "description": "Larvae (Chilo partellus) bore into stems causing dead hearts. Major pest in corn and sorghum causing 20-30% yield loss.",
                "symptoms": [
                    "Dead heart (central shoot dries and wilts)",
                    "Holes in stem with frass (excreta)",
                    "Frass at entry points",
                    "Stunted growth with side tillering"
                ],
                "treatments": {
                    "chemical": [
                        "Chlorpyrifos 20% EC @ 2.5ml/L",
                        "Cartap hydrochloride 4% G @ 10kg/ha (soil)",
                        "Fipronil 5% SC @ 1ml/L",
                        "Emamectin benzoate 5% SG @ 0.4g/L"
                    ],
                    "organic": [
                        "Trichogramma chilonis release @ 5 cards/ha",
                        "Neem seed kernel extract @ 5% (50g/L)",
                        "Light traps for moth monitoring (5/acre)",
                        "Bird perches @ 25/acre"
                    ],
                    "prevention": [
                        "Destroy crop residues after harvest",
                        "Intercrop with legumes (cowpea, soybean)",
                        "Use resistant varieties (e.g., Pusa Composite)",
                        "Early sowing to avoid peak pest (June-July)",
                        "Set up pheromone traps @ 10/ha"
                    ]
                }
            },
            "Healthy": {
                "severity": "Healthy",
                "description": "Healthy corn plant with bright green leaves and normal ear development.",
                "symptoms": [
                    "Bright green leaves with no spots",
                    "No pustules or lesions",
                    "Normal tassel development",
                    "Healthy ears with full kernel development"
                ],
                "treatments": {
                    "chemical": [
                        "No treatment needed",
                        "Preventive insecticide if borer history"
                    ],
                    "organic": [
                        "Foliar micronutrient spray (Zn, B)",
                        "Compost tea at knee-high stage",
                        "Vermiwash at tasseling"
                    ],
                    "prevention": [
                        "Proper spacing (60x20 cm)",
                        "Adequate irrigation at silking stage",
                        "Balanced NPK (120:60:40 kg/ha)",
                        "Weed control in first 45 days",
                        "Crop rotation with legumes"
                    ]
                }
            }
        }
    },
    "wheat": {
        "name": "Wheat",
        "icon": "🌾",
        "diseases": {
            "Yellow Rust": {
                "severity": "High",
                "description": "Yellow-orange pustules in stripes on leaves. Severe infection causes stunting and reduced grain fill. Cool (10-15°C) and humid conditions favor disease.",
                "symptoms": [
                    "Yellow-orange stripes of pustules",
                    "Pustule lines parallel to leaf veins",
                    "Leaf yellowing and drying",
                    "Stunted growth and poor tillering"
                ],
                "treatments": {
                    "chemical": [
                        "Propiconazole 25% EC @ 1ml/L",
                        "Tebuconazole 25% EC @ 1ml/L",
                        "Difenoconazole 25% EC @ 1ml/L",
                        "Triadimefon 25% WP @ 1g/L"
                    ],
                    "organic": [
                        "Neem oil spray @ 5ml/L",
                        "Compost tea foliar application",
                        "Sulfur spray (wettable sulfur 80% WP @ 3g/L)",
                        "Cow dung extract spray"
                    ],
                    "prevention": [
                        "Plant resistant varieties (HD-2967, DBW-187)",
                        "Early sowing (before Nov 15)",
                        "Adequate spacing (20 cm row spacing)",
                        "Monitor at flag leaf stage (Zadok 37-39)",
                        "Avoid excess nitrogen application"
                    ]
                }
            },
            "Karnal Bunt": {
                "severity": "Medium",
                "description": "Partially or fully replaced grains with black spore masses. Fishy odor. Reduces grain quality and export value.",
                "symptoms": [
                    "Black spore masses inside grains",
                    "Distinct fishy/rotten odor",
                    "Partially filled or empty grains",
                    "Reduced test weight and milling quality"
                ],
                "treatments": {
                    "chemical": [
                        "Carbendazim 50% WP @ 1g/L seed treatment",
                        "Propiconazole 25% EC @ 1ml/L foliar at booting",
                        "Tebuconazole 25% EC @ 1ml/L at flowering"
                    ],
                    "organic": [
                        "Hot water seed treatment (52°C for 10 min)",
                        "Trichoderma viride seed treatment @ 4g/kg",
                        "Mustard cake soil application",
                        "Neem cake @ 200kg/ha"
                    ],
                    "prevention": [
                        "Use certified seeds (toll-free quality)",
                        "Crop rotation (2 years with non-cereal)",
                        "Avoid late sowing (after Dec 15)",
                        "Resistant varieties (HD-2967)",
                        "Deep summer ploughing"
                    ]
                }
            },
            "Healthy": {
                "severity": "Healthy",
                "description": "Healthy wheat plant with normal green tillers and developing grain heads.",
                "symptoms": [
                    "Green healthy leaves with no spots",
                    "Normal tillering (8-12 tillers/plant)",
                    "No rust pustules or powdery coating",
                    "Healthy grain heads with full kernels"
                ],
                "treatments": {
                    "chemical": [
                        "Preventive propiconazole at flag leaf stage",
                        "Seed treatment with carbedazim @ 2g/kg"
                    ],
                    "organic": [
                        "Compost tea foliar spray at tillering",
                        "Micronutrient spray (Zn, Fe) at booting",
                        "Vermiwash application"
                    ],
                    "prevention": [
                        "Certified seeds from authorized source",
                        "Crop rotation (2-year gap)",
                        "Balanced NPK (120:60:40 kg/ha)",
                        "Timely sowing (Nov 1-15)",
                        "Proper irrigation at crown root and flowering"
                    ]
                }
            }
        }
    },
    "rice": {
        "name": "Rice",
        "icon": "🍚",
        "diseases": {
            "Rice Blast": {
                "severity": "High",
                "description": "Diamond-shaped lesions on leaves with gray centers and brown borders. Neck blast causes panicle breakage (white head). Most destructive rice disease.",
                "symptoms": [
                    "Diamond-shaped lesions on leaves",
                    "Gray centers with brown borders",
                    "Neck rot at panicle base (neck blast)",
                    "Panicle breakage (white heads)"
                ],
                "treatments": {
                    "chemical": [
                        "Tricyclazole 75% WP @ 1g/L",
                        "Carbendazim 50% WP @ 1g/L",
                        "Isoprothiolane 40% EC @ 1.5ml/L",
                        "Edifenphos 50% EC @ 1ml/L"
                    ],
                    "organic": [
                        "Neem cake application @ 200kg/ha at basal",
                        "Compost tea foliar spray",
                        "Silicon fertilizer (silica) @ 100kg/ha",
                        "Cow dung ash spray"
                    ],
                    "prevention": [
                        "Use resistant varieties (Pusa Basmati-1, IR-64)",
                        "Balanced nitrogen (avoid excess - split doses)",
                        "Proper water management (5 cm standing water)",
                        "Remove weed hosts (barnyard grass)",
                        "Avoid dense planting (20x15 cm)"
                    ]
                }
            },
            "Bacterial Leaf Blight": {
                "severity": "High",
                "description": "Yellow to white lesions along leaf margins. Wilting of seedlings (Kresek). Severe yield loss in susceptible varieties.",
                "symptoms": [
                    "Yellow-white lesions on leaf margins",
                    "Wilted seedlings (Kresek symptom)",
                    "Empty grains and chaffy panicles",
                    "Bacterial ooze on cut stems"
                ],
                "treatments": {
                    "chemical": [
                        "Streptocycline 90% + Copper oxychloride @ 0.5g/L",
                        "Kasugamycin 3% SL @ 2ml/L",
                        "Copper oxychloride 50% WP @ 3g/L",
                        "Oxytetracycline 19.3% @ 1g/L"
                    ],
                    "organic": [
                        "Compost tea with Pseudomonas fluorescens",
                        "Neem cake soil application @ 200kg/ha",
                        "Cow urine spray (1:10 dilution)",
                        "Biofertilizer Azotobacter + PSB"
                    ],
                    "prevention": [
                        "Use resistant varieties (Pusa-44, MTU-7029)",
                        "Avoid excessive nitrogen",
                        "Drain field periodically (alternate wetting-drying)",
                        "Hot water seed treatment (52°C, 30 min)",
                        "Avoid planting in infected fields"
                    ]
                }
            },
            "Healthy": {
                "severity": "Healthy",
                "description": "Healthy rice plant with normal green tillers and developing panicles.",
                "symptoms": [
                    "Dark green healthy leaves",
                    "Normal tillering (15-20 tillers/hill)",
                    "No lesions or spots",
                    "Healthy panicles with full grains"
                ],
                "treatments": {
                    "chemical": [
                        "Preventive tricyclazole if blast history",
                        "Seed treatment with carbendazim @ 2g/kg"
                    ],
                    "organic": [
                        "Silicon supplementation @ 100kg/ha",
                        "Compost tea at tillering stage",
                        "Azolla incorporation for N and Si"
                    ],
                    "prevention": [
                        "Balanced fertilization (100:50:50 NPK)",
                        "Proper water depth (5 cm during vegetative)",
                        "Weed control in first 30 days",
                        "Monitor at booting stage",
                        "Proper spacing (20x15 cm)"
                    ]
                }
            }
        }
    },
    "cotton": {
        "name": "Cotton",
        "icon": "🧵",
        "diseases": {
            "Leaf Curl Virus": {
                "severity": "High",
                "description": "Upward curling of leaves, vein thickening, and stunted growth. Spread by whiteflies (Bemisia tabaci). No cure available.",
                "symptoms": [
                    "Leaves curl upward distinctly",
                    "Vein thickening and darkening",
                    "Stunted plant growth",
                    "Reduced bolls and lint quality"
                ],
                "treatments": {
                    "chemical": [
                        "Imidacloprid 17.8% SL @ 0.5ml/L (whitefly)",
                        "Thiamethoxam 25% WG @ 0.3g/L",
                        "Buprofezin 25% SC @ 1ml/L",
                        "NOTE: No direct cure for virus - vector control only"
                    ],
                    "organic": [
                        "Neem oil @ 5ml/L for whitefly",
                        "Yellow sticky traps @ 10/acre",
                        "Reflective silver mulch",
                        "Release Encarsia formosa parasitoids"
                    ],
                    "prevention": [
                        "Use resistant varieties (RCH-650, Ankur-651)",
                        "Control whitefly vectors from seedling stage",
                        "Remove infected plants and burn immediately",
                        "Avoid planting near infected fields",
                        "Use 40-mesh insect-proof nets in nursery"
                    ]
                }
            },
            "Bollworm": {
                "severity": "High",
                "description": "Larvae (Helicoverpa armigera) bore into cotton bolls causing shedding and damage. Major pest causing 30-40% yield loss.",
                "symptoms": [
                    "Bored holes in bolls with frass",
                    "Shedding of squares and young bolls",
                    "Frass (excreta) at entry holes",
                    "Damaged lint with stained fibers"
                ],
                "treatments": {
                    "chemical": [
                        "Spinosad 45% SC @ 0.3ml/L",
                        "Emamectin benzoate 5% SG @ 0.4g/L",
                        "Chlorantraniliprole 18.5% SC @ 0.3ml/L",
                        "Indoxacarb 14.5% SC @ 0.5ml/L"
                    ],
                    "organic": [
                        "Trichogramma chilonis release @ 5 cards/ha",
                        "Neem seed kernel extract @ 5% (50g/L)",
                        "Pheromone traps (Helilure) @ 10/ha",
                        "HaNPV (Nuclear Polyhedrosis Virus) @ 250 LE/ha"
                    ],
                    "prevention": [
                        "Grow Bt cotton varieties (BG-II)",
                        "Regular monitoring (10 plants/acre)",
                        "Remove and destroy fruiting bodies with eggs",
                        "Avoid monocropping - rotate with legumes",
                        "Install bird perches @ 25/acre"
                    ]
                }
            },
            "Healthy": {
                "severity": "Healthy",
                "description": "Healthy cotton plant with normal green leaves and developing bolls.",
                "symptoms": [
                    "Green healthy leaves with no curling",
                    "Normal monopodial branching",
                    "Healthy squares and bolls",
                    "No signs of pest damage"
                ],
                "treatments": {
                    "chemical": [
                        "Preventive insecticide for whitefly at squaring",
                        "Acetamiprid 20% SP @ 0.3g/L"
                    ],
                    "organic": [
                        "Neem cake @ 200kg/ha at basal",
                        "Compost tea foliar spray",
                        "Vermicompost @ 5 tons/ha"
                    ],
                    "prevention": [
                        "Grow resistant Bt varieties",
                        "Whitefly monitoring with yellow traps",
                        "Timely sowing (April-May for kharif)",
                        "Proper spacing (75x30 cm or 120x60 cm)",
                        "Field sanitation and destruction of crop residues"
                    ]
                }
            }
        }
    },
    "soybean": {
        "name": "Soybean",
        "icon": "🫘",
        "diseases": {
            "Frogeye Leaf Spot": {
                "severity": "Medium",
                "description": "Small circular spots with gray centers and reddish-brown borders. Spots may have dark borders with light centers.",
                "symptoms": [
                    "Circular gray spots (2-5mm)",
                    "Reddish-brown borders around spots",
                    "Dark ring around spots in severe cases",
                    "Leaf yellowing and premature defoliation"
                ],
                "treatments": {
                    "chemical": [
                        "Azoxystrobin 23% SC @ 1ml/L",
                        "Pyraclostrobin 20% WG @ 1g/L",
                        "Mancozeb 75% WP @ 2g/L",
                        "Carbendazim 50% WP @ 1g/L"
                    ],
                    "organic": [
                        "Compost tea foliar spray",
                        "Neem oil @ 5ml/L with sticker",
                        "Copper fungicide (organic grade)",
                        "Baking soda spray (preventive)"
                    ],
                    "prevention": [
                        "Rotate with corn or wheat (2-year gap)",
                        "Plant resistant varieties (JS-335, MACS-450)",
                        "Remove and destroy crop residue",
                        "Avoid dense planting (45x10 cm)",
                        "Balanced fertilization (20:60:40 NPK)"
                    ]
                }
            },
            "Yellow Mosaic Virus": {
                "severity": "High",
                "description": "Yellow mosaic pattern on leaves. Stunted growth. Spread by whiteflies (Bemisia tabaci). No cure available.",
                "symptoms": [
                    "Yellow mosaic pattern on leaves",
                    "Stunted plant growth",
                    "Reduced pod formation",
                    "Leaf curling and thickening"
                ],
                "treatments": {
                    "chemical": [
                        "Imidacloprid 17.8% SL @ 0.5ml/L (whitefly)",
                        "Thiamethoxam 25% WG @ 0.3g/L",
                        "NOTE: No cure for virus - remove infected plants"
                    ],
                    "organic": [
                        "Neem oil @ 5ml/L for whitefly control",
                        "Yellow sticky traps @ 10/acre",
                        "Reflective mulch (silver)",
                        "Release Encarsia formosa"
                    ],
                    "prevention": [
                        "Use resistant varieties (JS-335, MACS-450)",
                        "Control whitefly vectors from seedling stage",
                        "Remove infected plants immediately",
                        "Avoid planting near infected fields",
                        "Use 40-mesh insect-proof nets"
                    ]
                }
            },
            "Healthy": {
                "severity": "Healthy",
                "description": "Healthy soybean plant with normal trifoliate green leaves and developing pods.",
                "symptoms": [
                    "Green trifoliate leaves with no spots",
                    "No spots, lesions, or mosaic",
                    "Normal pod development (30-40 pods/plant)",
                    "Healthy root nodules (pink interior)"
                ],
                "treatments": {
                    "chemical": [
                        "No treatment needed",
                        "Preventive fungicide if disease history"
                    ],
                    "organic": [
                        "Rhizobium inoculation @ 20g/kg seed",
                        "Compost tea at flowering",
                        "PSB (Phosphate Solubilizing Bacteria) @ 2kg/ha"
                    ],
                    "prevention": [
                        "Proper Rhizobium inoculation at sowing",
                        "Adequate spacing (45x10 cm)",
                        "Weed control in first 45 days",
                        "Monitor regularly for whiteflies",
                        "Crop rotation with cereals"
                    ]
                }
            }
        }
    },

    "apple": {
        "name": "Apple",
        "icon": "🍎",
        "diseases": {
            "Apple Scab": {
                "severity": "Medium",
                "description": "Fungal disease causing dark, scabby lesions on leaves and fruit, common in cool, wet spring weather.",
                "symptoms": ["Olive-green to dark brown spots on leaves", "Scabby, corky lesions on fruit surface", "Premature leaf and fruit drop"],
                "treatments": {
                    "chemical": ["Captan 50% WP @ 2g/L", "Myclobutanil 10% WP @ 1g/L"],
                    "organic": ["Sulfur-based fungicide spray", "Remove and destroy fallen leaves in autumn"],
                    "prevention": ["Prune for good air circulation", "Plant scab-resistant varieties", "Avoid overhead irrigation"]
                }
            },
            "Black Rot": {
                "severity": "High",
                "description": "Fungal disease causing leaf spots, fruit rot, and cankers on branches.",
                "symptoms": ["Purple-bordered leaf spots ('frogeye')", "Fruit rot starting at blossom end", "Sunken cankers on bark"],
                "treatments": {
                    "chemical": ["Captan 50% WP @ 2g/L", "Thiophanate-methyl 70% WP @ 1g/L"],
                    "organic": ["Prune out cankers and dead wood", "Copper-based fungicide spray"],
                    "prevention": ["Remove mummified fruit from tree and ground", "Prune during dry weather", "Maintain tree vigor with proper fertilization"]
                }
            },
            "Cedar Apple Rust": {
                "severity": "Medium",
                "description": "Fungal disease requiring both apple and cedar/juniper trees to complete its life cycle, causing bright orange leaf spots.",
                "symptoms": ["Bright yellow-orange spots on upper leaf surface", "Tube-like structures on leaf underside", "Premature defoliation in severe cases"],
                "treatments": {
                    "chemical": ["Myclobutanil 10% WP @ 1g/L", "Mancozeb 75% WP @ 2.5g/L"],
                    "organic": ["Remove nearby cedar/juniper trees if feasible", "Sulfur spray during early season"],
                    "prevention": ["Plant rust-resistant apple varieties", "Increase distance from cedar/juniper trees"]
                }
            },
            "Healthy": {"severity": "Healthy", "description": "No signs of disease detected.", "symptoms": ["Green, unblemished foliage"], "treatments": {"chemical": [], "organic": [], "prevention": ["Continue regular monitoring", "Maintain balanced fertilization"]}}
        }
    },

    "grape": {
        "name": "Grape",
        "icon": "🍇",
        "diseases": {
            "Black Rot": {
                "severity": "High",
                "description": "One of the most destructive grape diseases, causing fruit to shrivel into hard black 'mummies'.",
                "symptoms": ["Circular tan leaf spots with dark borders", "Fruit turns brown then shrivels/mummifies", "Small black fungal fruiting bodies on lesions"],
                "treatments": {
                    "chemical": ["Mancozeb 75% WP @ 2.5g/L", "Myclobutanil 10% WP @ 1g/L"],
                    "organic": ["Remove mummified berries and infected canes", "Copper-based fungicide spray"],
                    "prevention": ["Prune for canopy airflow", "Remove wild/abandoned grapevines nearby"]
                }
            },
            "Esca (Black Measles)": {
                "severity": "High",
                "description": "Complex fungal trunk disease causing tiger-stripe leaf patterns and internal wood decay.",
                "symptoms": ["Tiger-stripe pattern of yellow/red between leaf veins", "Dark spots on berries ('black measles')", "Sudden vine collapse possible in severe cases"],
                "treatments": {
                    "chemical": ["No fully effective chemical cure - trunk injection treatments exist commercially"],
                    "organic": ["Prune out and destroy infected wood", "Avoid pruning wounds during wet weather"],
                    "prevention": ["Use certified disease-free planting material", "Protect pruning cuts with wound sealant"]
                }
            },
            "Leaf Blight": {
                "severity": "Medium",
                "description": "Fungal disease (Isariopsis leaf spot) causing angular brown lesions on grape leaves.",
                "symptoms": ["Angular dark brown leaf lesions", "Lesions may merge causing large dead areas", "Premature defoliation"],
                "treatments": {
                    "chemical": ["Mancozeb 75% WP @ 2.5g/L", "Copper oxychloride 50% WP @ 3g/L"],
                    "organic": ["Improve canopy ventilation", "Remove heavily infected leaves"],
                    "prevention": ["Avoid overhead irrigation", "Prune for good air circulation"]
                }
            },
            "Healthy": {"severity": "Healthy", "description": "No signs of disease detected.", "symptoms": ["Green, unblemished foliage"], "treatments": {"chemical": [], "organic": [], "prevention": ["Continue regular monitoring", "Maintain balanced fertilization"]}}
        }
    },

    "cherry": {
        "name": "Cherry",
        "icon": "🍒",
        "diseases": {
            "Powdery Mildew": {
                "severity": "Medium",
                "description": "Fungal disease producing a white powdery coating on leaves and shoots, common in humid conditions with poor airflow.",
                "symptoms": ["White powdery patches on leaves and young shoots", "Leaf curling and distortion", "Stunted shoot growth"],
                "treatments": {
                    "chemical": ["Sulfur-based fungicide @ 2g/L", "Myclobutanil 10% WP @ 1g/L"],
                    "organic": ["Neem oil spray", "Improve air circulation via pruning"],
                    "prevention": ["Avoid excess nitrogen fertilization", "Prune for open canopy structure"]
                }
            },
            "Healthy": {"severity": "Healthy", "description": "No signs of disease detected.", "symptoms": ["Green, unblemished foliage"], "treatments": {"chemical": [], "organic": [], "prevention": ["Continue regular monitoring", "Maintain balanced fertilization"]}}
        }
    },

    "peach": {
        "name": "Peach",
        "icon": "🍑",
        "diseases": {
            "Bacterial Spot": {
                "severity": "Medium",
                "description": "Bacterial disease causing dark spots on leaves and fruit, worse in warm, wet weather.",
                "symptoms": ["Small dark angular spots on leaves, often with yellow halo", "Spots on fruit that can crack and scab", "Leaf shot-holing as spots fall out"],
                "treatments": {
                    "chemical": ["Copper oxychloride 50% WP @ 3g/L", "Oxytetracycline spray where permitted"],
                    "organic": ["Copper-based organic bactericide", "Avoid working in wet orchards to reduce spread"],
                    "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Prune for airflow"]
                }
            },
            "Healthy": {"severity": "Healthy", "description": "No signs of disease detected.", "symptoms": ["Green, unblemished foliage"], "treatments": {"chemical": [], "organic": [], "prevention": ["Continue regular monitoring", "Maintain balanced fertilization"]}}
        }
    },

    "orange": {
        "name": "Orange",
        "icon": "🍊",
        "diseases": {
            "Citrus Greening (HLB)": {
                "severity": "High",
                "description": "Serious bacterial disease spread by psyllid insects, causing yellowing, misshapen bitter fruit, and eventual tree decline. No cure currently exists.",
                "symptoms": ["Blotchy yellow mottling on leaves (asymmetric)", "Small, lopsided, bitter fruit that stays partly green", "Twig dieback and gradual tree decline"],
                "treatments": {
                    "chemical": ["Control psyllid vector with Imidacloprid 17.8% SL @ 0.5ml/L"],
                    "organic": ["Remove and destroy infected trees to prevent spread", "Yellow sticky traps for psyllid monitoring"],
                    "prevention": ["Use certified disease-free nursery stock", "Regular psyllid vector control", "Monitor and remove infected trees promptly"]
                }
            }
        }
    },

    "pepper": {
        "name": "Bell Pepper",
        "icon": "🫑",
        "diseases": {
            "Bacterial Spot": {
                "severity": "Medium",
                "description": "Bacterial disease causing dark, raised spots on leaves and fruit, spread by rain splash and contaminated seed.",
                "symptoms": ["Small dark water-soaked spots on leaves", "Raised, scabby spots on fruit", "Leaf yellowing and drop in severe cases"],
                "treatments": {
                    "chemical": ["Copper oxychloride 50% WP @ 3g/L", "Streptomycin sulfate spray where permitted"],
                    "organic": ["Copper-based organic bactericide", "Use disease-free certified seed"],
                    "prevention": ["Avoid overhead watering", "Rotate crops (avoid peppers/tomatoes in same spot yearly)", "Disinfect tools between plants"]
                }
            },
            "Healthy": {"severity": "Healthy", "description": "No signs of disease detected.", "symptoms": ["Green, unblemished foliage"], "treatments": {"chemical": [], "organic": [], "prevention": ["Continue regular monitoring", "Maintain balanced fertilization"]}}
        }
    },

    "squash": {
        "name": "Squash",
        "icon": "🎃",
        "diseases": {
            "Powdery Mildew": {
                "severity": "Medium",
                "description": "Common fungal disease producing white powdery patches on leaves, especially in warm, dry days with cool nights.",
                "symptoms": ["White powdery coating on upper and lower leaf surfaces", "Yellowing and browning of affected leaves", "Reduced fruit yield and quality"],
                "treatments": {
                    "chemical": ["Sulfur-based fungicide @ 2g/L", "Myclobutanil 10% WP @ 1g/L"],
                    "organic": ["Neem oil spray", "Milk spray solution (1:9 milk to water)"],
                    "prevention": ["Plant mildew-resistant varieties", "Space plants for airflow", "Avoid overhead watering"]
                }
            }
        }
    },

    "strawberry": {
        "name": "Strawberry",
        "icon": "🍓",
        "diseases": {
            "Leaf Scorch": {
                "severity": "Medium",
                "description": "Fungal disease causing purple-red blotches on leaves that can merge into large scorched-looking areas.",
                "symptoms": ["Small purple spots that enlarge and merge", "Leaves appear scorched/burned from a distance", "Reduced plant vigor and yield"],
                "treatments": {
                    "chemical": ["Captan 50% WP @ 2g/L", "Myclobutanil 10% WP @ 1g/L"],
                    "organic": ["Remove and destroy infected leaves", "Copper-based fungicide spray"],
                    "prevention": ["Space plants for good airflow", "Avoid overhead irrigation", "Remove old infected leaves after harvest"]
                }
            },
            "Healthy": {"severity": "Healthy", "description": "No signs of disease detected.", "symptoms": ["Green, unblemished foliage"], "treatments": {"chemical": [], "organic": [], "prevention": ["Continue regular monitoring", "Maintain balanced fertilization"]}}
        }
    },

    "raspberry": {
        "name": "Raspberry", "icon": "🍇",
        "diseases": {
            "Healthy": {"severity": "Healthy", "description": "No signs of disease detected.", "symptoms": ["Green, unblemished foliage"], "treatments": {"chemical": [], "organic": [], "prevention": ["Continue regular monitoring", "Maintain balanced fertilization"]}}
        }
    },

    "blueberry": {
        "name": "Blueberry", "icon": "🫐",
        "diseases": {
            "Healthy": {"severity": "Healthy", "description": "No signs of disease detected.", "symptoms": ["Green, unblemished foliage"], "treatments": {"chemical": [], "organic": [], "prevention": ["Continue regular monitoring", "Maintain balanced fertilization"]}}
        }
    }
}

# Market prices (per quintal in INR)
MARKET_DATA = {
    "wheat": {"price": 2400, "change": 1.2},
    "rice": {"price": 3200, "change": -0.5},
    "corn": {"price": 2100, "change": 0.8},
    "soybean": {"price": 4500, "change": 2.1},
    "cotton": {"price": 6200, "change": -1.2},
    "tomato": {"price": 2800, "change": 3.5},
    "potato": {"price": 1500, "change": -0.8},
    "onion": {"price": 3500, "change": 1.5},
    "sugarcane": {"price": 340, "change": 0.3},
    "mustard": {"price": 5200, "change": -0.4},
    "groundnut": {"price": 5800, "change": 1.8}
}

MANDI_PREMIUMS = {
    "national": 1.0,
    "delhi": 1.05,
    "mumbai": 1.10,
    "kolkata": 0.95,
    "chennai": 1.08,
    "hyderabad": 1.02,
    "ludhiana": 1.03,
    "kanpur": 0.98
}

CROP_ICONS = {
    "wheat": "🌾", "rice": "🍚", "corn": "🌽", "soybean": "🫘",
    "cotton": "🧵", "tomato": "🍅", "potato": "🥔", "onion": "🧅",
    "sugarcane": "🎋", "mustard": "🌿", "groundnut": "🥜"
}

CROP_NAMES = {
    "wheat": "Wheat", "rice": "Rice", "corn": "Corn",
    "soybean": "Soybean", "cotton": "Cotton", "tomato": "Tomato",
    "potato": "Potato", "onion": "Onion", "sugarcane": "Sugarcane",
    "mustard": "Mustard", "groundnut": "Groundnut"
}
