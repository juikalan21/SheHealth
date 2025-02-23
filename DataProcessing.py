import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

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

# Convert the JSON data to a pandas DataFrame
df = pd.DataFrame(data)

# Extract relevant features
features = []
for index, row in df.iterrows():
    features.append({
        'age': row['demographics']['age'],
        'bmi': row['healthMetrics']['bmi'],
        'systolic': int(row['healthMetrics']['bloodPressure'].split('/')[0]),
        'diastolic': int(row['healthMetrics']['bloodPressure'].split('/')[1]),
        'total_cholesterol': row['healthMetrics']['cholesterolLevels']['total'],
        'ldl': row['healthMetrics']['cholesterolLevels']['ldl'],
        'hdl': row['healthMetrics']['cholesterolLevels']['hdl'],
        'stress_level': 1 if row['lifestyle']['stressLevels'] == 'low' else 
                        2 if row['lifestyle']['stressLevels'] == 'medium' else 3
    })

features_df = pd.DataFrame(features)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

# Hierarchical CLustering Implementation
# Split the data into training and testing sets
X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)

# Hierarchical Clustering
# Create and visualize the dendrogram to determine the optimal number of clusters
plt.figure(figsize=(12, 8))
dendrogram = dendrogram(linkage(X_train, method='ward'))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.axhline(y=6, color='r', linestyle='--')  # Drawing a cutoff line
plt.show()

# Based on the dendrogram, choose the number of clusters (let's say 4)
n_clusters = 4
hierarchical_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
hierarchical_labels = hierarchical_model.fit_predict(X_train)

# Evaluate the hierarchical clustering
silhouette_hierarchical = silhouette_score(X_train, hierarchical_labels)
davies_bouldin_hierarchical = davies_bouldin_score(X_train, hierarchical_labels)

print(f"Hierarchical Clustering - Silhouette Score: {silhouette_hierarchical:.4f}")
print(f"Hierarchical Clustering - Davies-Bouldin Index: {davies_bouldin_hierarchical:.4f}")

# Visualize the clusters with PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

plt.figure(figsize=(10, 8))
for cluster in range(n_clusters):
    plt.scatter(X_train_pca[hierarchical_labels == cluster, 0], 
                X_train_pca[hierarchical_labels == cluster, 1],
                label=f'Cluster {cluster}')
plt.title('Hierarchical Clustering Results (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Gaussian Mixture Model Implementation
# Determine optimal number of components for GMM using BIC
n_components_range = range(1, 10)
bic_scores = []
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_train)
    bic_scores.append(gmm.bic(X_train))

# Plot BIC scores
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, marker='o')
plt.title('BIC Scores for Different Numbers of Components')
plt.xlabel('Number of Components')
plt.ylabel('BIC Score')
plt.grid(True)
plt.show()

# Choose the number of components with the lowest BIC score
optimal_n_components = n_components_range[np.argmin(bic_scores)]
print(f"Optimal number of components based on BIC: {optimal_n_components}")

# Train GMM with the optimal number of components
gmm_model = GaussianMixture(n_components=optimal_n_components, random_state=42)
gmm_model.fit(X_train)
gmm_labels = gmm_model.predict(X_train)

# Evaluate the GMM
silhouette_gmm = silhouette_score(X_train, gmm_labels)
davies_bouldin_gmm = davies_bouldin_score(X_train, gmm_labels)

print(f"GMM - Silhouette Score: {silhouette_gmm:.4f}")
print(f"GMM - Davies-Bouldin Index: {davies_bouldin_gmm:.4f}")

# Visualize the GMM clusters with PCA
plt.figure(figsize=(10, 8))
for cluster in range(optimal_n_components):
    plt.scatter(X_train_pca[gmm_labels == cluster, 0], 
                X_train_pca[gmm_labels == cluster, 1],
                label=f'Cluster {cluster}')
plt.title('GMM Clustering Results (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Visualize GMM cluster probabilities (feature of GMM - soft clustering)
probabilities = gmm_model.predict_proba(X_train)

plt.figure(figsize=(12, 8))
plt.imshow(probabilities[:25], aspect='auto', cmap='viridis')
plt.colorbar(label='Probability')
plt.xlabel('Cluster')
plt.ylabel('Sample Index')
plt.title('GMM Cluster Membership Probabilities (First 25 Samples)')
plt.show()

# Model Evaluation and Selection
# Test both models on the test set
hierarchical_test_labels = hierarchical_model.fit_predict(X_test)
gmm_test_labels = gmm_model.predict(X_test)

# Evaluate both models on the test set
silhouette_hierarchical_test = silhouette_score(X_test, hierarchical_test_labels)
davies_bouldin_hierarchical_test = davies_bouldin_score(X_test, hierarchical_test_labels)

silhouette_gmm_test = silhouette_score(X_test, gmm_test_labels)
davies_bouldin_gmm_test = davies_bouldin_score(X_test, gmm_test_labels)

print("Test Set Evaluation:")
print(f"Hierarchical Clustering - Silhouette Score: {silhouette_hierarchical_test:.4f}")
print(f"Hierarchical Clustering - Davies-Bouldin Index: {davies_bouldin_hierarchical_test:.4f}")
print(f"GMM - Silhouette Score: {silhouette_gmm_test:.4f}")
print(f"GMM - Davies-Bouldin Index: {davies_bouldin_gmm_test:.4f}")

# Compare models and select the best one
if (silhouette_gmm_test > silhouette_hierarchical_test and 
    davies_bouldin_gmm_test < davies_bouldin_hierarchical_test):
    print("GMM is the better model based on test metrics.")
    best_model = gmm_model
    best_labels = gmm_labels
    n_best_clusters = optimal_n_components
elif (silhouette_hierarchical_test > silhouette_gmm_test and 
      davies_bouldin_hierarchical_test < davies_bouldin_gmm_test):
    print("Hierarchical clustering is the better model based on test metrics.")
    best_model = hierarchical_model
    best_labels = hierarchical_labels
    n_best_clusters = n_clusters
else:
    print("Mixed results. Consider additional evaluation metrics or domain knowledge.")
    # Default to GMM for this example as it often handles complex health data better
    best_model = gmm_model
    best_labels = gmm_labels
    n_best_clusters = optimal_n_components

# Cluster Analysis and Recommendation Generation
# Apply the best model to the entire dataset
all_labels = best_model.fit_predict(scaled_features) if isinstance(best_model, AgglomerativeClustering) else best_model.predict(scaled_features)

# Add cluster labels to original dataframe
df['cluster'] = all_labels

# Analyze each cluster
cluster_profiles = []

for cluster in range(n_best_clusters):
    cluster_data = features_df[all_labels == cluster]
    
    profile = {
        'cluster_id': cluster,
        'size': len(cluster_data),
        'age_mean': cluster_data['age'].mean(),
        'age_range': f"{cluster_data['age'].min()}-{cluster_data['age'].max()}",
        'bmi_mean': cluster_data['bmi'].mean(),
        'blood_pressure_mean': f"{cluster_data['systolic'].mean():.1f}/{cluster_data['diastolic'].mean():.1f}",
        'cholesterol_mean': cluster_data['total_cholesterol'].mean(),
        'hdl_mean': cluster_data['hdl'].mean(),
        'ldl_mean': cluster_data['ldl'].mean(),
        'stress_level_mean': cluster_data['stress_level'].mean()
    }
    
    cluster_profiles.append(profile)

# Display cluster profiles
cluster_profiles_df = pd.DataFrame(cluster_profiles)
print("Cluster Profiles:")
print(cluster_profiles_df.to_string(index=False))

# Example of generating recommendations based on cluster profiles
def generate_recommendations(cluster_profiles):
    recommendations = {}
    
    for profile in cluster_profiles:
        cluster_id = profile['cluster_id']
        recommendations[cluster_id] = {
            'health_focus_areas': [],
            'lifestyle_recommendations': [],
            'monitoring_recommendations': []
        }
        
        # Age-based recommendations
        if profile['age_mean'] > 50:
            recommendations[cluster_id]['health_focus_areas'].append("Age-related preventive care")
            recommendations[cluster_id]['monitoring_recommendations'].append("Annual comprehensive health checkups")
        elif profile['age_mean'] < 30:
            recommendations[cluster_id]['health_focus_areas'].append("Reproductive health and wellness")
            
        # BMI-based recommendations
        if profile['bmi_mean'] > 30:
            recommendations[cluster_id]['health_focus_areas'].append("Weight management")
            recommendations[cluster_id]['lifestyle_recommendations'].append("Structured exercise program, 150 minutes per week")
            recommendations[cluster_id]['lifestyle_recommendations'].append("Balanced nutrition plan")
        elif profile['bmi_mean'] < 18.5:
            recommendations[cluster_id]['health_focus_areas'].append("Nutritional adequacy")
            
        # Blood pressure recommendations
        systolic = float(profile['blood_pressure_mean'].split('/')[0])
        diastolic = float(profile['blood_pressure_mean'].split('/')[1])
        
        if systolic > 130 or diastolic > 80:
            recommendations[cluster_id]['health_focus_areas'].append("Cardiovascular health")
            recommendations[cluster_id]['lifestyle_recommendations'].append("DASH diet approach")
            recommendations[cluster_id]['monitoring_recommendations'].append("Regular blood pressure monitoring")
            
        # Cholesterol recommendations
        if profile['cholesterol_mean'] > 200 or profile['ldl_mean'] > 130:
            recommendations[cluster_id]['health_focus_areas'].append("Cholesterol management")
            recommendations[cluster_id]['lifestyle_recommendations'].append("Heart-healthy diet rich in omega-3s")
            recommendations[cluster_id]['monitoring_recommendations'].append("Quarterly lipid panel testing")
            
        # Stress-based recommendations
        if profile['stress_level_mean'] > 2.5:
            recommendations[cluster_id]['health_focus_areas'].append("Stress management")
            recommendations[cluster_id]['lifestyle_recommendations'].append("Daily mindfulness practice")
            recommendations[cluster_id]['lifestyle_recommendations'].append("Sleep hygiene improvement")
            
    return recommendations

# Generate and display recommendations
cluster_recommendations = generate_recommendations(cluster_profiles)

for cluster_id, recs in cluster_recommendations.items():
    print(f"\nRecommendations for Cluster {cluster_id}:")
    print(f"  Health Focus Areas: {', '.join(recs['health_focus_areas'])}")
    print(f"  Lifestyle Recommendations:")
    for rec in recs['lifestyle_recommendations']:
        print(f"    - {rec}")
    print(f"  Monitoring Recommendations:")
    for rec in recs['monitoring_recommendations']:
        print(f"    - {rec}")

# Cluster visualization Dashboard
# Function to create a comprehensive cluster visualization dashboard
def visualize_clusters(df, features_df, scaled_features, all_labels, n_clusters):
    plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 2)
    
    # 1. PCA visualization of clusters
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(scaled_features)
    
    ax1 = plt.subplot(gs[0, :])
    for cluster in range(n_clusters):
        ax1.scatter(data_pca[all_labels == cluster, 0], 
                   data_pca[all_labels == cluster, 1],
                   label=f'Cluster {cluster}', s=50, alpha=0.7)
    ax1.set_title('Cluster Visualization (PCA)', fontsize=16)
    ax1.set_xlabel('Principal Component 1', fontsize=12)
    ax1.set_ylabel('Principal Component 2', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Age distribution by cluster
    ax2 = plt.subplot(gs[1, 0])
    sns.boxplot(x=all_labels, y=features_df['age'], ax=ax2)
    ax2.set_title('Age Distribution by Cluster', fontsize=16)
    ax2.set_xlabel('Cluster', fontsize=12)
    ax2.set_ylabel('Age', fontsize=12)
    
    # 3. BMI distribution by cluster
    ax3 = plt.subplot(gs[1, 1])
    sns.boxplot(x=all_labels, y=features_df['bmi'], ax=ax3)
    ax3.set_title('BMI Distribution by Cluster', fontsize=16)
    ax3.set_xlabel('Cluster', fontsize=12)
    ax3.set_ylabel('BMI', fontsize=12)
    
    # 4. Cholesterol profiles by cluster
    ax4 = plt.subplot(gs[2, 0])
    cluster_chol = features_df.groupby(all_labels)['total_cholesterol'].mean()
    cluster_hdl = features_df.groupby(all_labels)['hdl'].mean()
    cluster_ldl = features_df.groupby(all_labels)['ldl'].mean()
    
    x = np.arange(n_clusters)
    width = 0.25
    
    ax4.bar(x - width, cluster_chol, width, label='Total Cholesterol')
    ax4.bar(x, cluster_hdl, width, label='HDL')
    ax4.bar(x + width, cluster_ldl, width, label='LDL')
    
    ax4.set_title('Cholesterol Profiles by Cluster', fontsize=16)
    ax4.set_xlabel('Cluster', fontsize=12)
    ax4.set_ylabel('Level (mg/dL)', fontsize=12)
    ax4.set_xticks(x)
    ax4.legend()
    
    # 5. Blood pressure by cluster
    ax5 = plt.subplot(gs[2, 1])
    cluster_systolic = features_df.groupby(all_labels)['systolic'].mean()
    cluster_diastolic = features_df.groupby(all_labels)['diastolic'].mean()
    
    ax5.bar(x - width/2, cluster_systolic, width, label='Systolic')
    ax5.bar(x + width/2, cluster_diastolic, width, label='Diastolic')
    
    ax5.set_title('Blood Pressure by Cluster', fontsize=16)
    ax5.set_xlabel('Cluster', fontsize=12)
    ax5.set_ylabel('mmHg', fontsize=12)
    ax5.set_xticks(x)
    ax5.legend()
    
    # 6. Correlations between features within each cluster
    for i, cluster in enumerate(range(n_clusters)):
        ax = plt.subplot(gs[3, i] if i < 2 else None)
        if ax:
            cluster_data = features_df[all_labels == cluster]
            corr = cluster_data.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                       annot=True, square=True, linewidths=.5, ax=ax)
            ax.set_title(f'Feature Correlations: Cluster {cluster}', fontsize=16)
    
    plt.tight_layout()
    plt.show()

# Call the function to create the dashboard
visualize_clusters(df, features_df, scaled_features, all_labels, n_best_clusters)

# Personalized Recommendation System
def assign_to_cluster_and_recommend(new_patient_data, model, scaler, cluster_recommendations):
    """
    Assign a new patient to a cluster and provide personalized recommendations.
    
    Parameters:
    new_patient_data (dict): Dictionary containing patient health data
    model: Trained clustering model (either Hierarchical or GMM)
    scaler: Fitted StandardScaler from training
    cluster_recommendations (dict): Recommendations mapped to each cluster
    
    Returns:
    dict: Patient's cluster assignment and personalized recommendations
    """
    # Extract features from new patient data (same format as training data)
    features = {
        'age': new_patient_data['demographics']['age'],
        'bmi': new_patient_data['healthMetrics']['bmi'],
        'systolic': int(new_patient_data['healthMetrics']['bloodPressure'].split('/')[0]),
        'diastolic': int(new_patient_data['healthMetrics']['bloodPressure'].split('/')[1]),
        'total_cholesterol': new_patient_data['healthMetrics']['cholesterolLevels']['total'],
        'ldl': new_patient_data['healthMetrics']['cholesterolLevels']['ldl'],
        'hdl': new_patient_data['healthMetrics']['cholesterolLevels']['hdl'],
        'stress_level': 1 if new_patient_data['lifestyle']['stressLevels'] == 'low' else 
                        2 if new_patient_data['lifestyle']['stressLevels'] == 'medium' else 3
    }
    
    # Convert to numpy array and scale
    feature_array = np.array([list(features.values())])
    scaled_features = scaler.transform(feature_array)
    
    # Predict cluster
    if isinstance(model, GaussianMixture):
        cluster_id = model.predict(scaled_features)[0]
        # Get cluster probabilities (unique to GMM)
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = probabilities[cluster_id]
    else:
        cluster_id = model.fit_predict(scaled_features)[0]
        confidence = None
    
    # Get recommendations for the assigned cluster
    recommendations = cluster_recommendations[cluster_id]
    
    # Create personalized output with patient-specific nuances
    result = {
        'patient_id': new_patient_data['id'],
        'assigned_cluster': int(cluster_id),
        'cluster_confidence': float(confidence) if confidence is not None else None,
        'personalized_recommendations': {
            'health_focus_areas': recommendations['health_focus_areas'].copy(),
            'lifestyle_recommendations': recommendations['lifestyle_recommendations'].copy(),
            'monitoring_recommendations': recommendations['monitoring_recommendations'].copy()
        }
    }
    
    # Personalize based on patient's specific conditions
    conditions = new_patient_data['medicalHistory']['chronicConditions']
    
    # Add condition-specific recommendations
    for condition in conditions:
        if condition == "hypertension" and "Cardiovascular health" not in result['personalized_recommendations']['health_focus_areas']:
            result['personalized_recommendations']['health_focus_areas'].append("Cardiovascular health")
            result['personalized_recommendations']['monitoring_recommendations'].append("Weekly blood pressure monitoring")
        
        elif condition == "type 2 diabetes" and "Blood sugar management" not in result['personalized_recommendations']['health_focus_areas']:
            result['personalized_recommendations']['health_focus_areas'].append("Blood sugar management")
            result['personalized_recommendations']['lifestyle_recommendations'].append("Carbohydrate counting and glycemic index awareness")
            result['personalized_recommendations']['monitoring_recommendations'].append("Daily glucose monitoring")
        
        elif "anxiety" in condition.lower() and "Mental wellness" not in result['personalized_recommendations']['health_focus_areas']:
            result['personalized_recommendations']['health_focus_areas'].append("Mental wellness")
            result['personalized_recommendations']['lifestyle_recommendations'].append("Daily mindfulness practice")
    
    return result

# Example usage with a new patient
new_patient = data[0]  # Using the first patient as an example of a new patient
patient_recommendation = assign_to_cluster_and_recommend(
    new_patient, 
    best_model, 
    scaler, 
    cluster_recommendations
)

# Display personalized recommendations
print(f"Personalized Recommendations for Patient ID: {patient_recommendation['patient_id']}")
print(f"Assigned to Cluster: {patient_recommendation['assigned_cluster']}")
if patient_recommendation['cluster_confidence'] is not None:
    print(f"Cluster Assignment Confidence: {patient_recommendation['cluster_confidence']:.2f}")

print("\nHealth Focus Areas:")
for area in patient_recommendation['personalized_recommendations']['health_focus_areas']:
    print(f"  - {area}")

print("\nLifestyle Recommendations:")
for rec in patient_recommendation['personalized_recommendations']['lifestyle_recommendations']:
    print(f"  - {rec}")

print("\nMonitoring Recommendations:")
for rec in patient_recommendation['personalized_recommendations']['monitoring_recommendations']:
    print(f"  - {rec}")