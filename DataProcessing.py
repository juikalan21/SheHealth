import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Dataset
data = [
  {
    "id": 1,
    "demographics": {
      "age": 34,
      "location": "Seattle, WA",
      "ethnicity": "Asian"
    },
    "healthMetrics": {
      "bmi": 23.1,
      "bloodPressure": "118/75",
      "cholesterolLevels": {
        "total": 180,
        "ldl": 100,
        "hdl": 65
      },
      "menstrualCycleData": {
        "cycleLength": 28,
        "lastPeriod": "2025-01-29",
        "flowIntensity": "moderate"
      }
    },
    "lifestyle": {
      "diet": "vegetarian",
      "exerciseHabits": "moderate activity 4x weekly",
      "sleepPatterns": "7.5 hours average",
      "stressLevels": "medium"
    },
    "preferences": {
      "communicationChannels": ["email", "app notifications"],
      "healthGoals": ["stress reduction", "better sleep quality"]
    },
    "medicalHistory": {
      "chronicConditions": ["seasonal allergies"],
      "medications": ["loratadine PRN"],
      "allergies": ["pollen", "penicillin"]
    }
  },
  {
    "id": 2,
    "demographics": {
      "age": 56,
      "location": "Chicago, IL",
      "ethnicity": "African American"
    },
    "healthMetrics": {
      "bmi": 28.4,
      "bloodPressure": "138/85",
      "cholesterolLevels": {
        "total": 210,
        "ldl": 130,
        "hdl": 45
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-05",
        "flowIntensity": "varies"
      }
    },
    "lifestyle": {
      "diet": "omnivore",
      "exerciseHabits": "light walking 3x weekly",
      "sleepPatterns": "6 hours average, interrupted",
      "stressLevels": "high"
    },
    "preferences": {
      "communicationChannels": ["phone", "SMS"],
      "healthGoals": ["blood pressure management", "weight loss"]
    },
    "medicalHistory": {
      "chronicConditions": ["hypertension", "type 2 diabetes"],
      "medications": ["lisinopril 10mg", "metformin 500mg"],
      "allergies": ["sulfa drugs"]
    }
  },
  {
    "id": 3,
    "demographics": {
      "age": 21,
      "location": "Austin, TX",
      "ethnicity": "Hispanic"
    },
    "healthMetrics": {
      "bmi": 20.8,
      "bloodPressure": "110/70",
      "cholesterolLevels": {
        "total": 165,
        "ldl": 85,
        "hdl": 68
      },
      "menstrualCycleData": {
        "cycleLength": 30,
        "lastPeriod": "2025-02-05",
        "flowIntensity": "light"
      }
    },
    "lifestyle": {
      "diet": "flexitarian",
      "exerciseHabits": "runs 5x weekly, yoga 2x weekly",
      "sleepPatterns": "8 hours average",
      "stressLevels": "low"
    },
    "preferences": {
      "communicationChannels": ["app notifications", "email"],
      "healthGoals": ["maintain fitness", "nutrition optimization"]
    },
    "medicalHistory": {
      "chronicConditions": [],
      "medications": ["multivitamin"],
      "allergies": ["none"]
    }
  },
  {
    "id": 4,
    "demographics": {
      "age": 42,
      "location": "Portland, OR",
      "ethnicity": "White"
    },
    "healthMetrics": {
      "bmi": 24.7,
      "bloodPressure": "122/78",
      "cholesterolLevels": {
        "total": 195,
        "ldl": 110,
        "hdl": 55
      },
      "menstrualCycleData": {
        "cycleLength": 26,
        "lastPeriod": "2025-02-10",
        "flowIntensity": "heavy"
      }
    },
    "lifestyle": {
      "diet": "pescatarian",
      "exerciseHabits": "cycling 3x weekly, strength training 2x weekly",
      "sleepPatterns": "7 hours average",
      "stressLevels": "medium"
    },
    "preferences": {
      "communicationChannels": ["email", "SMS"],
      "healthGoals": ["stress management", "hormonal balance"]
    },
    "medicalHistory": {
      "chronicConditions": ["endometriosis", "migraines"],
      "medications": ["sumatriptan PRN", "birth control pill"],
      "allergies": ["latex"]
    }
  },
  {
    "id": 5,
    "demographics": {
      "age": 68,
      "location": "Miami, FL",
      "ethnicity": "Cuban American"
    },
    "healthMetrics": {
      "bmi": 27.2,
      "bloodPressure": "142/88",
      "cholesterolLevels": {
        "total": 230,
        "ldl": 150,
        "hdl": 42
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-05",
        "flowIntensity": "varies"
      }
    },
    "lifestyle": {
      "diet": "Mediterranean",
      "exerciseHabits": "swimming 2x weekly, light walking daily",
      "sleepPatterns": "6.5 hours average",
      "stressLevels": "medium"
    },
    "preferences": {
      "communicationChannels": ["phone", "mail"],
      "healthGoals": ["heart health", "mobility maintenance"]
    },
    "medicalHistory": {
      "chronicConditions": ["hypertension", "osteoarthritis"],
      "medications": ["amlodipine 5mg", "acetaminophen PRN"],
      "allergies": ["ibuprofen"]
    }
  },
  {
    "id": 6,
    "demographics": {
      "age": 29,
      "location": "Denver, CO",
      "ethnicity": "Mixed race"
    },
    "healthMetrics": {
      "bmi": 21.5,
      "bloodPressure": "115/72",
      "cholesterolLevels": {
        "total": 175,
        "ldl": 90,
        "hdl": 70
      },
      "menstrualCycleData": {
        "cycleLength": 28,
        "lastPeriod": "2025-01-25",
        "flowIntensity": "moderate"
      }
    },
    "lifestyle": {
      "diet": "plant-based",
      "exerciseHabits": "hiking 2x weekly, rock climbing 1x weekly",
      "sleepPatterns": "8 hours average",
      "stressLevels": "low"
    },
    "preferences": {
      "communicationChannels": ["app notifications", "SMS"],
      "healthGoals": ["athletic performance", "mental wellness"]
    },
    "medicalHistory": {
      "chronicConditions": ["anxiety"],
      "medications": ["escitalopram 10mg"],
      "allergies": ["shellfish"]
    }
  },
  {
    "id": 7,
    "demographics": {
      "age": 51,
      "location": "Philadelphia, PA",
      "ethnicity": "Italian American"
    },
    "healthMetrics": {
      "bmi": 29.8,
      "bloodPressure": "130/85",
      "cholesterolLevels": {
        "total": 220,
        "ldl": 140,
        "hdl": 48
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2024-12-15",
        "flowIntensity": "light"
      }
    },
    "lifestyle": {
      "diet": "omnivore, high carb",
      "exerciseHabits": "weight training 2x weekly",
      "sleepPatterns": "6 hours average",
      "stressLevels": "high"
    },
    "preferences": {
      "communicationChannels": ["email", "phone"],
      "healthGoals": ["weight loss", "menopause symptom management"]
    },
    "medicalHistory": {
      "chronicConditions": ["GERD", "prediabetes"],
      "medications": ["omeprazole 20mg"],
      "allergies": ["dust mites"]
    }
  },
  {
    "id": 8,
    "demographics": {
      "age": 37,
      "location": "Minneapolis, MN",
      "ethnicity": "White"
    },
    "healthMetrics": {
      "bmi": 26.1,
      "bloodPressure": "124/80",
      "cholesterolLevels": {
        "total": 190,
        "ldl": 105,
        "hdl": 58
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-05",
        "flowIntensity": "varies"
      }
    },
    "lifestyle": {
      "diet": "paleo",
      "exerciseHabits": "CrossFit 4x weekly",
      "sleepPatterns": "7 hours average",
      "stressLevels": "medium"
    },
    "preferences": {
      "communicationChannels": ["SMS", "app notifications"],
      "healthGoals": ["muscle gain", "endurance"]
    },
    "medicalHistory": {
      "chronicConditions": ["asthma"],
      "medications": ["albuterol inhaler PRN"],
      "allergies": ["cats", "amoxicillin"]
    }
  },
  {
    "id": 9,
    "demographics": {
      "age": 24,
      "location": "San Diego, CA",
      "ethnicity": "Filipino"
    },
    "healthMetrics": {
      "bmi": 22.3,
      "bloodPressure": "112/68",
      "cholesterolLevels": {
        "total": 170,
        "ldl": 85,
        "hdl": 72
      },
      "menstrualCycleData": {
        "cycleLength": 32,
        "lastPeriod": "2025-01-18",
        "flowIntensity": "moderate"
      }
    },
    "lifestyle": {
      "diet": "pescatarian",
      "exerciseHabits": "surfing 3x weekly, yoga 2x weekly",
      "sleepPatterns": "7.5 hours average",
      "stressLevels": "low"
    },
    "preferences": {
      "communicationChannels": ["email", "app notifications"],
      "healthGoals": ["general wellness", "stress management"]
    },
    "medicalHistory": {
      "chronicConditions": ["eczema"],
      "medications": ["topical steroid cream PRN"],
      "allergies": ["nickel"]
    }
  },
  {
    "id": 10,
    "demographics": {
      "age": 63,
      "location": "Boston, MA",
      "ethnicity": "Irish American"
    },
    "healthMetrics": {
      "bmi": 25.8,
      "bloodPressure": "135/84",
      "cholesterolLevels": {
        "total": 205,
        "ldl": 125,
        "hdl": 50
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-05",
        "flowIntensity": "varies"
      }
    },
    "lifestyle": {
      "diet": "low sodium",
      "exerciseHabits": "walking daily, tai chi 2x weekly",
      "sleepPatterns": "6.5 hours average, early riser",
      "stressLevels": "medium"
    },
    "preferences": {
      "communicationChannels": ["phone", "mail"],
      "healthGoals": ["heart health", "joint mobility"]
    },
    "medicalHistory": {
      "chronicConditions": ["hypertension", "osteoporosis"],
      "medications": ["alendronate weekly", "losartan 25mg"],
      "allergies": ["sulfa drugs"]
    }
  },
  {
    "id": 11,
    "demographics": {
      "age": 32,
      "location": "Nashville, TN",
      "ethnicity": "White"
    },
    "healthMetrics": {
      "bmi": 24.0,
      "bloodPressure": "118/76",
      "cholesterolLevels": {
        "total": 185,
        "ldl": 105,
        "hdl": 60
      },
      "menstrualCycleData": {
        "cycleLength": 29,
        "lastPeriod": "2025-02-08",
        "flowIntensity": "heavy"
      }
    },
    "lifestyle": {
      "diet": "omnivore, home-cooked meals",
      "exerciseHabits": "jogging 3x weekly, dance class 1x weekly",
      "sleepPatterns": "7 hours average",
      "stressLevels": "medium"
    },
    "preferences": {
      "communicationChannels": ["SMS", "email"],
      "healthGoals": ["fertility planning", "stress reduction"]
    },
    "medicalHistory": {
      "chronicConditions": ["PCOS"],
      "medications": ["metformin 500mg", "prenatal vitamins"],
      "allergies": ["none"]
    }
  },
  {
    "id": 12,
    "demographics": {
      "age": 47,
      "location": "Atlanta, GA",
      "ethnicity": "African American"
    },
    "healthMetrics": {
      "bmi": 30.2,
      "bloodPressure": "140/88",
      "cholesterolLevels": {
        "total": 215,
        "ldl": 135,
        "hdl": 45
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-05",
        "flowIntensity": "varies"
      }
    },
    "lifestyle": {
      "diet": "southern traditional",
      "exerciseHabits": "stationary bike 2x weekly",
      "sleepPatterns": "5.5 hours average, shift worker",
      "stressLevels": "high"
    },
    "preferences": {
      "communicationChannels": ["phone", "SMS"],
      "healthGoals": ["weight loss", "better sleep"]
    },
    "medicalHistory": {
      "chronicConditions": ["sleep apnea", "hypertension"],
      "medications": ["lisinopril 20mg", "CPAP therapy"],
      "allergies": ["latex"]
    }
  },
  {
    "id": 13,
    "demographics": {
      "age": 19,
      "location": "Madison, WI",
      "ethnicity": "White"
    },
    "healthMetrics": {
      "bmi": 21.0,
      "bloodPressure": "110/68",
      "cholesterolLevels": {
        "total": 160,
        "ldl": 80,
        "hdl": 65
      },
      "menstrualCycleData": {
        "cycleLength": 28,
        "lastPeriod": "2025-02-02",
        "flowIntensity": "light"
      }
    },
    "lifestyle": {
      "diet": "vegetarian, college dining hall",
      "exerciseHabits": "intramural sports 2x weekly, walks to class",
      "sleepPatterns": "irregular, 6-9 hours",
      "stressLevels": "high during exams, medium otherwise"
    },
    "preferences": {
      "communicationChannels": ["app notifications", "SMS"],
      "healthGoals": ["mental health", "balanced nutrition"]
    },
    "medicalHistory": {
      "chronicConditions": ["seasonal allergies"],
      "medications": ["cetirizine PRN"],
      "allergies": ["pollen"]
    }
  },
  {
    "id": 14,
    "demographics": {
      "age": 58,
      "location": "Phoenix, AZ",
      "ethnicity": "Native American"
    },
    "healthMetrics": {
      "bmi": 32.1,
      "bloodPressure": "145/90",
      "cholesterolLevels": {
        "total": 240,
        "ldl": 155,
        "hdl": 40
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-05",
        "flowIntensity": "varies"
      }
    },
    "lifestyle": {
      "diet": "traditional, high carb",
      "exerciseHabits": "gardening, occasional walking",
      "sleepPatterns": "6 hours average",
      "stressLevels": "medium"
    },
    "preferences": {
      "communicationChannels": ["phone", "mail"],
      "healthGoals": ["diabetes management", "weight loss"]
    },
    "medicalHistory": {
      "chronicConditions": ["type 2 diabetes", "hypertension"],
      "medications": ["metformin 1000mg", "lisinopril 20mg", "atorvastatin 40mg"],
      "allergies": ["penicillin"]
    }
  },
  {
    "id": 15,
    "demographics": {
      "age": 28,
      "location": "Brooklyn, NY",
      "ethnicity": "Jewish American"
    },
    "healthMetrics": {
      "bmi": 20.5,
      "bloodPressure": "112/72",
      "cholesterolLevels": {
        "total": 165,
        "ldl": 85,
        "hdl": 65
      },
      "menstrualCycleData": {
        "cycleLength": 30,
        "lastPeriod": "2025-01-30",
        "flowIntensity": "moderate"
      }
    },
    "lifestyle": {
      "diet": "kosher, mostly vegetarian",
      "exerciseHabits": "running 3x weekly, HIIT 2x weekly",
      "sleepPatterns": "7 hours average",
      "stressLevels": "high"
    },
    "preferences": {
      "communicationChannels": ["email", "app notifications"],
      "healthGoals": ["anxiety management", "athletic performance"]
    },
    "medicalHistory": {
      "chronicConditions": ["generalized anxiety disorder"],
      "medications": ["buspirone 10mg"],
      "allergies": ["none"]
    }
  },
  {
    "id": 16,
    "demographics": {
      "age": 71,
      "location": "St. Petersburg, FL",
      "ethnicity": "White"
    },
    "healthMetrics": {
      "bmi": 26.8,
      "bloodPressure": "138/82",
      "cholesterolLevels": {
        "total": 210,
        "ldl": 130,
        "hdl": 45
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-05",
        "flowIntensity": "varies"
      }
    },
    "lifestyle": {
      "diet": "low sugar, heart healthy",
      "exerciseHabits": "water aerobics 3x weekly, walking daily",
      "sleepPatterns": "7 hours average, early riser",
      "stressLevels": "low"
    },
    "preferences": {
      "communicationChannels": ["phone", "printed materials"],
      "healthGoals": ["maintain independence", "heart health"]
    },
    "medicalHistory": {
      "chronicConditions": ["atrial fibrillation", "osteoarthritis"],
      "medications": ["warfarin 5mg", "acetaminophen PRN"],
      "allergies": ["contrast dye"]
    }
  },
  {
    "id": 17,
    "demographics": {
      "age": 33,
      "location": "Seattle, WA",
      "ethnicity": "Chinese American"
    },
    "healthMetrics": {
      "bmi": 22.0,
      "bloodPressure": "115/70",
      "cholesterolLevels": {
        "total": 175,
        "ldl": 90,
        "hdl": 70
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-05",
        "flowIntensity": "varies"
      }
    },
    "lifestyle": {
      "diet": "balanced omnivore, low processed foods",
      "exerciseHabits": "climbing 2x weekly, cycling commuter",
      "sleepPatterns": "7.5 hours average",
      "stressLevels": "medium"
    },
    "preferences": {
      "communicationChannels": ["email", "app notifications"],
      "healthGoals": ["work-life balance", "strength building"]
    },
    "medicalHistory": {
      "chronicConditions": ["asthma"],
      "medications": ["fluticasone inhaler"],
      "allergies": ["dust mites", "pollen"]
    }
  },
  {
    "id": 18,
    "demographics": {
      "age": 45,
      "location": "Dallas, TX",
      "ethnicity": "White"
    },
    "healthMetrics": {
      "bmi": 31.0,
      "bloodPressure": "132/86",
      "cholesterolLevels": {
        "total": 225,
        "ldl": 140,
        "hdl": 42
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-15",
        "flowIntensity": "light"
      }
    },
    "lifestyle": {
      "diet": "omnivore, frequent dining out",
      "exerciseHabits": "golf 1x weekly",
      "sleepPatterns": "6 hours average",
      "stressLevels": "high"
    },
    "preferences": {
      "communicationChannels": ["phone", "email"],
      "healthGoals": ["weight loss", "stress management"]
    },
    "medicalHistory": {
      "chronicConditions": ["high cholesterol", "migraines"],
      "medications": ["sumatriptan PRN", "atorvastatin 20mg"],
      "allergies": ["sulfa drugs"]
    }
  },
  {
    "id": 19,
    "demographics": {
      "age": 26,
      "location": "Boulder, CO",
      "ethnicity": "Mixed race"
    },
    "healthMetrics": {
      "bmi": 21.2,
      "bloodPressure": "108/65",
      "cholesterolLevels": {
        "total": 155,
        "ldl": 75,
        "hdl": 68
      },
      "menstrualCycleData": {
        "cycleLength": 28,
        "lastPeriod": "2025-02-12",
        "flowIntensity": "moderate"
      }
    },
    "lifestyle": {
      "diet": "vegan, whole foods",
      "exerciseHabits": "trail running 4x weekly, yoga daily",
      "sleepPatterns": "8 hours average",
      "stressLevels": "low"
    },
    "preferences": {
      "communicationChannels": ["app notifications", "email"],
      "healthGoals": ["athletic performance", "holistic wellness"]
    },
    "medicalHistory": {
      "chronicConditions": ["none"],
      "medications": ["vitamin B12 supplement"],
      "allergies": ["none"]
    }
  },
  {
    "id": 20,
    "demographics": {
      "age": 52,
      "location": "Detroit, MI",
      "ethnicity": "African American"
    },
    "healthMetrics": {
      "bmi": 29.5,
      "bloodPressure": "142/90",
      "cholesterolLevels": {
        "total": 230,
        "ldl": 145,
        "hdl": 40
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-05",
        "flowIntensity": "varies"
      }
    },
    "lifestyle": {
      "diet": "traditional soul food",
      "exerciseHabits": "walking 2x weekly",
      "sleepPatterns": "6 hours average, shift worker",
      "stressLevels": "high"
    },
    "preferences": {
      "communicationChannels": ["SMS", "phone"],
      "healthGoals": ["blood pressure management", "weight management"]
    },
    "medicalHistory": {
      "chronicConditions": ["hypertension", "gout"],
      "medications": ["hydrochlorothiazide 25mg", "allopurinol 300mg"],
      "allergies": ["aspirin"]
    }
  },
  {
    "id": 21,
    "demographics": {
      "age": 39,
      "location": "Portland, OR",
      "ethnicity": "White"
    },
    "healthMetrics": {
      "bmi": 24.5,
      "bloodPressure": "120/78",
      "cholesterolLevels": {
        "total": 190,
        "ldl": 110,
        "hdl": 55
      },
      "menstrualCycleData": {
        "cycleLength": 27,
        "lastPeriod": "2025-02-01",
        "flowIntensity": "light to moderate"
      }
    },
    "lifestyle": {
      "diet": "pescatarian, local produce",
      "exerciseHabits": "cycling 3x weekly, backpacking monthly",
      "sleepPatterns": "7.5 hours average",
      "stressLevels": "medium"
    },
    "preferences": {
      "communicationChannels": ["email", "app notifications"],
      "healthGoals": ["fertility planning", "endurance building"]
    },
    "medicalHistory": {
      "chronicConditions": ["mild depression"],
      "medications": ["fluoxetine 20mg"],
      "allergies": ["none"]
    }
  },
  {
    "id": 22,
    "demographics": {
      "age": 65,
      "location": "Tucson, AZ",
      "ethnicity": "Mexican American"
    },
    "healthMetrics": {
      "bmi": 27.5,
      "bloodPressure": "138/84",
      "cholesterolLevels": {
        "total": 210,
        "ldl": 130,
        "hdl": 48
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-05",
        "flowIntensity": "varies"
      }
    },
    "lifestyle": {
      "diet": "traditional Mexican, modified for diabetes",
      "exerciseHabits": "walking daily, gardening",
      "sleepPatterns": "6.5 hours average",
      "stressLevels": "medium"
    },
    "preferences": {
      "communicationChannels": ["phone", "family member assistance"],
      "healthGoals": ["diabetes management", "maintain independence"]
    },
    "medicalHistory": {
      "chronicConditions": ["type 2 diabetes", "cataracts"],
      "medications": ["metformin 1000mg", "glipizide 10mg"],
      "allergies": ["codeine"]
    }
  },
  {
    "id": 23,
    "demographics": {
      "age": 22,
      "location": "Minneapolis, MN",
      "ethnicity": "Somali American"
    },
    "healthMetrics": {
      "bmi": 20.1,
      "bloodPressure": "110/65",
      "cholesterolLevels": {
        "total": 160,
        "ldl": 80,
        "hdl": 70
      },
      "menstrualCycleData": {
        "cycleLength": 30,
        "lastPeriod": "2025-01-20",
        "flowIntensity": "moderate"
      }
    },
    "lifestyle": {
      "diet": "halal, home-cooked meals",
      "exerciseHabits": "basketball 2x weekly, gym 3x weekly",
      "sleepPatterns": "7 hours average",
      "stressLevels": "medium"
    },
    "preferences": {
      "communicationChannels": ["SMS", "app notifications"],
      "healthGoals": ["muscle building", "mental health"]
    },
    "medicalHistory": {
      "chronicConditions": ["vitamin D deficiency"],
      "medications": ["vitamin D supplement"],
      "allergies": ["none"]
    }
  },
  {
    "id": 24,
    "demographics": {
      "age": 48,
      "location": "Charlotte, NC",
      "ethnicity": "White"
    },
    "healthMetrics": {
      "bmi": 26.8,
      "bloodPressure": "128/82",
      "cholesterolLevels": {
        "total": 205,
        "ldl": 125,
        "hdl": 50
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-05",
        "flowIntensity": "varies"
      }
    },
    "lifestyle": {
      "diet": "low carb",
      "exerciseHabits": "tennis 2x weekly, pilates 1x weekly",
      "sleepPatterns": "6 hours average, night sweats",
      "stressLevels": "high"
    },
    "preferences": {
      "communicationChannels": ["email", "phone"],
      "healthGoals": ["perimenopause symptom management", "weight maintenance"]
    },
    "medicalHistory": {
      "chronicConditions": ["hypothyroidism", "migraines"],
      "medications": ["levothyroxine 88mcg", "sumatriptan PRN"],
      "allergies": ["shellfish"]
    }
  },
  {
    "id": 25,
    "demographics": {
      "age": 31,
      "location": "Kansas City, MO",
      "ethnicity": "White"
    },
    "healthMetrics": {
      "bmi": 33.5,
      "bloodPressure": "130/85",
      "cholesterolLevels": {
        "total": 215,
        "ldl": 140,
        "hdl": 42
      },
      "menstrualCycleData": {
        "cycleLength": "irregular",
        "lastPeriod": "2025-01-05",
        "flowIntensity": "varies"
      }
    },
    "lifestyle": {
      "diet": "standard American, fast food frequent",
      "exerciseHabits": "sedentary",
      "sleepPatterns": "irregular, 5-7 hours",
      "stressLevels": "high"
    },
    "preferences": {
      "communicationChannels": ["SMS", "email"],
      "healthGoals": ["weight loss", "better sleep"]
    },
    "medicalHistory": {
      "chronicConditions": ["sleep apnea", "fatty liver disease"],
      "medications": ["CPAP therapy"],
      "allergies": ["ragweed"]
    }
  }
]

# Converting JSON to DataFrame
df = pd.json_normalize(data, sep='_')

# Selecting relevant features for clustering
features = [
    'demographics_age',
    'healthMetrics_bmi',
    'healthMetrics_cholesterolLevels_total',
    'healthMetrics_cholesterolLevels_ldl',
    'healthMetrics_cholesterolLevels_hdl',
    'healthMetrics_menstrualCycleData_cycleLength',
    'lifestyle_stressLevels'  # Categorical feature
]

# Extract selected features
df_selected = df[features].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Convert cycle length to numeric, handling 'irregular' values
df_selected.loc[:, 'healthMetrics_menstrualCycleData_cycleLength'] = pd.to_numeric(
    df_selected['healthMetrics_menstrualCycleData_cycleLength'].replace('irregular', np.nan), 
    errors='coerce'
)

# Fill NaN values with mean of column (for irregular cycle lengths)
cycle_mean = df_selected['healthMetrics_menstrualCycleData_cycleLength'].mean()
df_selected.loc[:, 'healthMetrics_menstrualCycleData_cycleLength'] = df_selected['healthMetrics_menstrualCycleData_cycleLength'].fillna(cycle_mean)

# Handle categorical data (e.g., stressLevels)
stress_mapping = {
    'low': 0,
    'medium': 1,
    'high': 2
}
# Extract more complex stress level strings and map to basic levels
df_selected.loc[:, 'lifestyle_stressLevels'] = df_selected['lifestyle_stressLevels'].apply(
    lambda x: 'high' if 'high' in str(x).lower() else 
             ('low' if 'low' in str(x).lower() else 'medium')
).map(stress_mapping)

# Normalize numerical data
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df_selected)

# Hierarchical Clustering
def hierarchical_clustering(data, n_clusters=3):
    linked = linkage(data, 'ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig('hierarchical_dendrogram.png')
    plt.show()

    # Fit Agglomerative Clustering
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    return cluster.fit_predict(data)

# Gaussian Mixture Models (GMM)
def gmm_clustering(data, n_clusters=3):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gmm.fit_predict(data)

# Apply clustering
n_clusters = 3  # Number of clusters (you can adjust this)
hierarchical_labels = hierarchical_clustering(df_normalized, n_clusters)
gmm_labels = gmm_clustering(df_normalized, n_clusters)

# Add cluster labels to the original DataFrame
df['hierarchical_cluster'] = hierarchical_labels
df['gmm_cluster'] = gmm_labels

# Display the results
print(df[['id', 'hierarchical_cluster', 'gmm_cluster']])

# Visualize clusters using PCA for dimensionality reduction
def visualize_clusters(data, labels, title, filename):
    # Reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    
    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Get unique clusters
    clusters = pca_df['Cluster'].unique()
    
    # Plot each cluster with a different color
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for i, cluster in enumerate(clusters):
        plt.scatter(
            pca_df[pca_df['Cluster'] == cluster]['PC1'],
            pca_df[pca_df['Cluster'] == cluster]['PC2'],
            s=100, 
            alpha=0.8,
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=f'Cluster {cluster}'
        )
    
    # Add annotations for data points
    for i, row in pca_df.iterrows():
        plt.annotate(
            f"{df.iloc[i]['id']}", 
            (row['PC1'] + 0.1, row['PC2'] + 0.1),
            fontsize=8
        )
    
    plt.title(title, fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title="Clusters")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Feature importance for clusters
def analyze_cluster_features(data, labels):
    # Group by cluster and calculate mean for each feature
    cluster_df = pd.DataFrame(data, columns=df_selected.columns)
    cluster_df['Cluster'] = labels
    
    # Calculate mean values per cluster
    cluster_means = cluster_df.groupby('Cluster').mean()
    
    # Create heatmap for feature importance per cluster
    plt.figure(figsize=(12, 8))
    plt.pcolor(cluster_means, cmap='coolwarm')
    plt.colorbar()
    plt.yticks(np.arange(0.5, len(cluster_means.index)), cluster_means.index)
    plt.xticks(np.arange(0.5, len(cluster_means.columns)), cluster_means.columns, rotation=90)
    plt.title('Feature Values by Cluster')
    plt.tight_layout()
    plt.savefig('cluster_feature_heatmap.png')
    plt.show()
    
    return cluster_means

# Visualize clusters
visualize_clusters(df_normalized, hierarchical_labels, 'Hierarchical Clustering Results', 'hierarchical_clusters.png')
visualize_clusters(df_normalized, gmm_labels, 'GMM Clustering Results', 'gmm_clusters.png')

# Analyze feature importance for hierarchical clusters
hierarchical_cluster_features = analyze_cluster_features(df_normalized, hierarchical_labels)
print("\nHierarchical Clustering - Feature Means by Cluster:")
print(hierarchical_cluster_features)

# Additional 3D visualization if there are enough samples
if len(df) > 10:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use the first 3 principal components
    pca = PCA(n_components=3)
    pca_3d = pca.fit_transform(df_normalized)
    
    # Plot 3D scatter
    for i in range(n_clusters):
        cluster_points = pca_3d[hierarchical_labels == i]
        ax.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1], 
            cluster_points[:, 2],
            s=80,
            alpha=0.7,
            label=f'Cluster {i}'
        )
    
    ax.set_title('3D Visualization of Hierarchical Clusters')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    plt.tight_layout()
    plt.savefig('3d_clusters.png')
    plt.show()

# Correlation matrix visualization
plt.figure(figsize=(12, 10))
correlation = df_selected.corr()
im = plt.imshow(correlation, cmap='coolwarm')
plt.colorbar(im)
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()