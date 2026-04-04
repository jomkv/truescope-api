import asyncio
from services.entity_extraction_service import EntityExtractionService

def test_ner_suite():
    service = EntityExtractionService()
    
    test_cases = [
        # --- 1. PERSON NAMES (Lowercase/Mixed) ---
        {"text": "donald trump is against marriage equality", "expect": "Donald Trump"},
        {"text": "vladimir putin visited beijing", "expect": "Vladimir Putin"},
        {"text": "taylor swift concert in singapore", "expect": "Taylor Swift"},
        {"text": "elon musk buys twitter", "expect": "Elon Musk"},
        {"text": "vp sara duterte says she will not resign", "expect": "Sara Duterte"},
        {"text": "bongbong marcos attends summit", "expect": "Marcos"},
        {"text": "lisa manoban spotted in paris", "expect": "Lisa Manoban"},
        {"text": "jack ma reappears in hangzhou", "expect": "Jack Ma"},
        {"text": "xi jinping meets biden", "expect": "Xi Jinping"},
        {"text": "joe biden speech on economy", "expect": "Joe Biden"},
        {"text": "kamala harris visits manila", "expect": "Kamala Harris"},
        {"text": "rizal was a national hero", "expect": "Rizal"},
        {"text": "catriona gray wins miss universe", "expect": "Catriona Gray"},
        {"text": "manny pacquiao training for fight", "expect": "Manny Pacquiao"},
        {"text": "hidilyn diaz wins gold", "expect": "Hidilyn Diaz"},
        {"text": "jose mari chan starts singing", "expect": "Jose Mari Chan"},
        {"text": "joshua garcia signs new contract", "expect": "Joshua Garcia"},
        {"text": "maricel soriano in new movie", "expect": "Maricel Soriano"},
        {"text": "sharon cuneta concert tour", "expect": "Sharon Cuneta"},
        {"text": "regine velasquez performs live", "expect": "Regine Velasquez"},
        {"text": "piolo pascual joins marathon", "expect": "Piolo Pascual"},
        {"text": "kathryn bernardo and daniel padilla", "expect": "Kathryn Bernardo"},
        {"text": "vice ganda on it's showtime", "expect": "Vice Ganda"},
        {"text": "anne curtis marathon in london", "expect": "Anne Curtis"},
        {"text": "marian rivera new teleserye", "expect": "Marian Rivera"},
        
        # --- 2. WEATHER & DISASTERS ---
        {"text": "super typhoon uwan caused great damage", "expect": "Uwan"},
        {"text": "tropical storm kristine approach luzon", "expect": "Kristine"},
        {"text": "bagyong uwan signals hoisted in manila", "expect": "Uwan"},
        {"text": "earthquake hit davao city province", "expect": "Davao City"},
        {"text": "volcano eruption in albay bicol", "expect": "Albay"},
        {"text": "flooding in cagayan valley region", "expect": "Cagayan Valley"},
        {"text": "landslide in benguet mountains", "expect": "Benguet"},
        {"text": "storm surge in leyte island", "expect": "Leyte"},
        {"text": "typhoon pepito leaves philipines", "expect": "Pepito"},
        {"text": "super typhoon super typhoon uwan", "expect": "Uwan"}, # Multi-generic test
        
        # --- 3. GEOPOLITICAL / PLACES ---
        {"text": "china and philippines dispute over scarborough shoal", "expect": "Scarborough Shoal"},
        {"text": "west philippine sea tension increases", "expect": "West Philippine Sea"},
        {"text": "sabah claim resurrected by sultanate", "expect": "Sabah"},
        {"text": "marcos visits malacanang palace", "expect": "Malacanang"},
        {"text": "meeting in asean summit cambodia", "expect": "Asean"},
        {"text": "united nations report on human rights", "expect": "United Nations"},
        {"text": "european union trade agreement", "expect": "European Union"},
        {"text": "south china sea freedom of navigation", "expect": "South China Sea"},
        {"text": "batanes island military base", "expect": "Batanes"},
        {"text": "spratly islands constructed runway", "expect": "Spratly Islands"},
        
        # --- 4. ORGANIZATIONS / BRANDS ---
        {"text": "apple inc launches new iphone", "expect": "Apple"},
        {"text": "samsung electronics profit surge", "expect": "Samsung"},
        {"text": "google ai development in mountain view", "expect": "Google"},
        {"text": "microsoft buy activision blizzard", "expect": "Microsoft"},
        {"text": "tesla factory in shanghai", "expect": "Tesla"},
        {"text": "san miguel corporation food prices", "expect": "San Miguel"},
        {"text": "jollibee opening in new york", "expect": "Jollibee"},
        {"text": "grab philippines service expansion", "expect": "Grab"},
        {"text": "shopee flash sale today", "expect": "Shopee"},
        {"text": "lazada voucher for electronics", "expect": "Lazada"},
        
        # --- 5. SOCIAL MEDIA / DEBUNKS ---
        {"text": "tiktok video shows fake bagyong kristine", "expect": "Kristine"},
        {"text": "facebook post about sara duterte resignation", "expect": "Sara Duterte"},
        {"text": "manipulated images of kamala harris", "expect": "Kamala Harris"},
        {"text": "youtube livestream of eruption", "expect": "Eruption"},
        {"text": "twitter hashgtag #MarcosOut", "expect": "Marcos"},
        {"text": "deepfake video of biden", "expect": "Biden"},
        {"text": "fact check on election results", "expect_fallback": True}, # Generic
        {"text": "debunking the military takeover rumor", "expect_fallback": True},
        
        # --- 6. GENERIC / NOISY (SHOULD BE EMPTY) ---
        {"text": "100 percent of the people said no", "expect_fallback": True},
        {"text": "the for and of in on", "expect_fallback": True},
        {"text": "saying that he says it is reported", "expect_fallback": True},
        {"text": "claiming that they claimed the claim", "expect_fallback": True},
        {"text": "super typhoon and tropical storm", "expect_fallback": True},
        {"text": "president vice senator senator", "expect_fallback": True},
        {"text": "city province municipality region", "expect_fallback": True},
        {"text": "it is what it is actually", "expect_fallback": True},
        {"text": "for more information check the report", "expect_fallback": True},
        {"text": "the end of the year results", "expect_fallback": True},
        {"text": "actually no one knows", "expect_fallback": True},
        {"text": "some say yes others say no", "expect_fallback": True},
        {"text": "only 10 percent of them", "expect_fallback": True},
        {"text": "highly credible reported source says", "expect_fallback": True},
        {"text": "breaking news update now", "expect_fallback": True},
        
        # --- 7. COMPLEX / MIXED ---
        {"text": "Marcos and Duterte join the APEC summit in USA", "expect": "Marcos"},
        {"text": "Typhoon Kristine heads to Batanes and Ilocos Norte", "expect": "Kristine"},
        {"text": "QC Mayor Joy Belmonte opens new park", "expect": "Belmonte"},
        {"text": "SM Mall of Asia fireworks display", "expect": "Mall of Asia"},
        {"text": "UP Diliman student protest", "expect": "UP Diliman"},
        {"text": "De La Salle vs Ateneo basketball game", "expect": "Ateneo"},
        {"text": "Boracay beach cleanup drive", "expect": "Boracay"},
        {"text": "NAIA Terminal 3 flight delay", "expect": "NAIA"},
        {"text": "EDSA traffic jam today", "expect": "EDSA"},
        {"text": "Makati City business district fire", "expect": "Makati"},
        {"text": "Cebu Pacific promo fare", "expect": "Cebu Pacific"},
        {"text": "Philippine Airlines flight to Japan", "expect": "Japan"},
        {"text": "Shopee vs Lazada comparison", "expect": "Lazada"},
        {"text": "BDO bank holiday notice", "expect": "BDO"},
        {"text": "Gcash outage report tonight", "expect": "Gcash"},
        {"text": "Paymaya wallet transfer error", "expect": "Paymaya"},
        {"text": "Maya bank savings rate", "expect": "Maya"},
        {"text": "Globe vs Smart internet speed", "expect": "Globe"},
        {"text": "Dito Telecommunity network coverage", "expect": "Dito"},
        {"text": "Sky Cable maintenance schedule", "expect": "Sky Cable"}
    ]

    print(f"=== NER STRESS TEST ({len(test_cases)} CASES) ===")
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        text = case["text"]
        entities = service.extract_entities(text)
        ent_texts = [e[0].lower() for e in entities]
        
        print(f"\n[{i:03d}] Input: '{text}'")
        print(f"      Extracted: {entities}")
        
        success = False
        if "expect" in case:
            expected = case["expect"].lower()
            success = any(expected.lower() in et.lower() for et in ent_texts)
        elif case.get("expect_fallback"):
            # Success if it returns the fallback (MISC label)
            success = any(e[1] == "MISC" for e in entities)
            
        if success:
            print("      Result: [PASS]")
            passed += 1
        else:
            print(f"      Result: [FAIL] (Expected: '{case.get('expect', 'Empty List')}')")

    print(f"\n{'='*30}")
    print(f"TOTAL PASSED: {passed}/{total} ({ (passed/total)*100:.1f}%)")
    print(f"{'='*30}")

if __name__ == "__main__":
    test_ner_suite()
