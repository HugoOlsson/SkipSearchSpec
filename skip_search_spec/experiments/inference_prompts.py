INFERENCE_TEST_PROMPTS_HARD = [
    # -------------------------------------------------------------------------
    # Factual / encyclopedic continuation
    # -------------------------------------------------------------------------
    (
        "encyclopedic_biology_photosynthesis",
        (
            "Article: Photosynthesis\n\n"
            "Photosynthesis is the process by which plants, algae, and some bacteria "
            "convert light energy into chemical energy. In green plants, this process "
            "takes place mainly in the leaves, where chlorophyll absorbs sunlight. "
            "The process uses carbon dioxide from the air and water from the soil to"
        ),
    ),
    (
        "encyclopedic_history_printing_press",
        (
            "Article: The printing press\n\n"
            "The printing press changed the way information moved through Europe. "
            "Before movable type became widespread, books were copied by hand or "
            "produced slowly by specialized workshops. The new technology made it "
            "possible to produce many copies of the same text, which"
        ),
    ),
    (
        "encyclopedic_geography_rivers",
        (
            "Article: Major rivers\n\n"
            "Rivers shape landscapes, support agriculture, and connect inland regions "
            "with seas and oceans. A river system usually includes a main channel, "
            "tributaries, floodplains, and a drainage basin. Over long periods of time,"
        ),
    ),
    (
        "encyclopedic_space_moon",
        (
            "Article: The Moon\n\n"
            "The Moon is Earth's only natural satellite and one of the brightest objects "
            "in the night sky. Its surface is covered with craters, plains, mountains, "
            "and fine dust. Because the Moon is close to Earth compared with other "
            "celestial bodies,"
        ),
    ),

    # -------------------------------------------------------------------------
    # Summarization-like prompts
    # -------------------------------------------------------------------------
    (
        "summary_renewable_energy",
        (
            "Passage:\n"
            "Many cities are investing in renewable energy projects to reduce their "
            "dependence on fossil fuels. Solar panels are being installed on public "
            "buildings, wind power contracts are being signed for municipal facilities, "
            "and older streetlights are being replaced with efficient LED systems. "
            "Although these projects require upfront spending, city officials argue "
            "that they can reduce long-term energy costs and improve air quality.\n\n"
            "Summary:\n"
        ),
    ),
    (
        "summary_school_lunch",
        (
            "Passage:\n"
            "A school district introduced a new lunch program after parents and teachers "
            "raised concerns about nutrition. The program adds more fresh vegetables, "
            "whole grains, and fruit while reducing heavily processed foods. Students "
            "were invited to taste several meals before the final menu was selected. "
            "The district plans to review participation numbers at the end of the year.\n\n"
            "Summary:\n"
        ),
    ),
    (
        "summary_library_renovation",
        (
            "Passage:\n"
            "The central library will close for six months while crews renovate the "
            "building. The project includes repairing the roof, updating the heating "
            "system, adding study rooms, and improving accessibility at the main entrance. "
            "During the closure, a temporary branch will operate from a nearby community "
            "center with limited hours and a smaller book collection.\n\n"
            "Summary:\n"
        ),
    ),
    (
        "summary_farmers_market",
        (
            "Passage:\n"
            "The weekly farmers market has grown steadily since it opened three years "
            "ago. Local growers sell vegetables, bread, eggs, honey, and flowers, while "
            "musicians perform near the entrance. The market manager said the event has "
            "helped small farms reach new customers and has brought more visitors to "
            "nearby shops on Saturday mornings.\n\n"
            "Summary:\n"
        ),
    ),

    # -------------------------------------------------------------------------
    # Question answering / exam-style completions
    # -------------------------------------------------------------------------
    (
        "qa_water_cycle",
        (
            "Science question:\n"
            "Explain the main stages of the water cycle and describe how water returns "
            "from the atmosphere to the surface of the Earth.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "qa_democracy",
        (
            "Civics question:\n"
            "What is the purpose of holding regular elections in a democratic system?\n\n"
            "Answer:\n"
        ),
    ),
    (
        "qa_supply_demand",
        (
            "Economics question:\n"
            "Explain how supply and demand can affect the price of a product in a market.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "qa_volcanoes",
        (
            "Earth science question:\n"
            "Why do many volcanoes form near the boundaries of tectonic plates?\n\n"
            "Answer:\n"
        ),
    ),

    # -------------------------------------------------------------------------
    # Math word problems, GSM8K-ish but self-contained
    # -------------------------------------------------------------------------
    (
        "math_apples",
        (
            "Problem:\n"
            "Maya has 18 apples. She gives 5 apples to her neighbor and then buys "
            "12 more apples at the market. How many apples does Maya have now?\n\n"
            "Solution:\n"
        ),
    ),
    (
        "math_train_tickets",
        (
            "Problem:\n"
            "A train ticket costs 7 dollars. A family buys 4 tickets and also pays "
            "6 dollars for parking. What is the total cost?\n\n"
            "Solution:\n"
        ),
    ),
    (
        "math_boxes",
        (
            "Problem:\n"
            "A warehouse has 9 shelves. Each shelf holds 8 boxes. Workers remove "
            "17 boxes for delivery. How many boxes remain in the warehouse?\n\n"
            "Solution:\n"
        ),
    ),
    (
        "math_recipe",
        (
            "Problem:\n"
            "A recipe uses 3 cups of flour for one cake. Lena wants to bake 5 cakes, "
            "but she already has 4 cups of flour at home. How many more cups of flour "
            "does she need?\n\n"
            "Solution:\n"
        ),
    ),

    # -------------------------------------------------------------------------
    # Code completion, HumanEval/MBPP-ish style
    # -------------------------------------------------------------------------
    (
        "code_python_reverse_words",
        (
            "def reverse_words(text: str) -> str:\n"
            "    \"\"\"Return a string with the words in reverse order.\n"
            "\n"
            "    Example:\n"
            "    reverse_words('red green blue') == 'blue green red'\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "code_python_count_vowels",
        (
            "def count_vowels(text: str) -> int:\n"
            "    \"\"\"Count how many vowels appear in the input string.\n"
            "\n"
            "    The vowels are a, e, i, o, and u. The function should treat uppercase\n"
            "    and lowercase letters the same way.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "code_python_merge_dicts",
        (
            "def merge_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:\n"
            "    \"\"\"Merge two dictionaries of counts.\n"
            "\n"
            "    If the same key appears in both dictionaries, the returned dictionary\n"
            "    should contain the sum of the two counts.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "code_python_is_palindrome",
        (
            "def is_palindrome(text: str) -> bool:\n"
            "    \"\"\"Return True if text is a palindrome after ignoring spaces and case.\n"
            "\n"
            "    Example:\n"
            "    is_palindrome('Never odd or even') == True\n"
            "    \"\"\"\n"
        ),
    ),

    # -------------------------------------------------------------------------
    # Translation-like / transformation prompts
    # -------------------------------------------------------------------------
    (
        "transform_formal_email",
        (
            "Original note:\n"
            "hey, i can't make it to the meeting today. can we move it to tomorrow?\n\n"
            "Formal version:\n"
        ),
    ),
    (
        "transform_bullet_to_paragraph",
        (
            "Notes:\n"
            "- The town opened a new park.\n"
            "- The park has walking paths, benches, and a playground.\n"
            "- Local families attended the opening ceremony.\n\n"
            "Paragraph:\n"
        ),
    ),
    (
        "transform_active_voice",
        (
            "Sentence:\n"
            "The report was reviewed by the committee before the announcement was made.\n\n"
            "Rewritten in active voice:\n"
        ),
    ),
    (
        "transform_simple_explanation",
        (
            "Technical sentence:\n"
            "Evaporation occurs when molecules at the surface of a liquid gain enough "
            "energy to enter the gas phase.\n\n"
            "Simple explanation:\n"
        ),
    ),

    # -------------------------------------------------------------------------
    # Creative but bounded prose
    # -------------------------------------------------------------------------
    (
        "story_lighthouse",
        (
            "Short story opening:\n\n"
            "The old lighthouse stood at the edge of the island, its white walls marked "
            "by years of salt and wind. Every evening, Nora climbed the spiral staircase "
            "to check the lamp before sunset. One autumn night, just as the fog began "
            "to roll across the water,"
        ),
    ),
    (
        "story_baker",
        (
            "Short story opening:\n\n"
            "Jonas opened the bakery before sunrise, as he had done every morning for "
            "twenty years. The first loaves were already cooling on the wooden rack, "
            "and the smell of cinnamon filled the narrow street outside. When he unlocked "
            "the front door,"
        ),
    ),
    (
        "story_robot_garden",
        (
            "Short story opening:\n\n"
            "The small robot had been built to water the garden, trim the vines, and "
            "measure the soil after rain. For months it followed the same routine in "
            "the quiet greenhouse. Then, one morning, it found a strange silver seed "
            "beside the tomato plants,"
        ),
    ),
    (
        "story_mountain_village",
        (
            "Short story opening:\n\n"
            "In a mountain village surrounded by pine forests, the winter festival began "
            "when the first bell rang from the old stone tower. Children carried lanterns "
            "through the snow while families prepared warm bread and soup. At the center "
            "of the square,"
        ),
    ),

    # -------------------------------------------------------------------------
    # Procedural / instructional text
    # -------------------------------------------------------------------------
    (
        "procedure_plant_seedlings",
        (
            "Guide: How to start vegetable seedlings indoors\n\n"
            "Starting seedlings indoors gives young plants a protected place to grow "
            "before they are moved outside. First, choose clean containers with drainage "
            "holes and fill them with seed-starting mix. Next,"
        ),
    ),
    (
        "procedure_bike_tire",
        (
            "Guide: How to fix a flat bicycle tire\n\n"
            "A flat tire can usually be repaired with a few simple tools. Turn the bike "
            "upside down or place it on a repair stand. Release the brake if needed, "
            "remove the wheel, and"
        ),
    ),
    (
        "procedure_emergency_kit",
        (
            "Guide: Preparing a basic emergency kit\n\n"
            "A basic emergency kit should contain supplies that help a household manage "
            "short disruptions in power, water, or transportation. Useful items include "
            "bottled water, shelf-stable food, a flashlight, batteries,"
        ),
    ),
    (
        "procedure_clean_workspace",
        (
            "Guide: Organizing a small workspace\n\n"
            "A small workspace is easier to use when each item has a clear place. Begin "
            "by removing objects that do not belong on the desk. Sort papers into a few "
            "simple categories, then"
        ),
    ),

    # -------------------------------------------------------------------------
    # Comparison / analysis completions
    # -------------------------------------------------------------------------
    (
        "compare_city_country",
        (
            "Comparison:\n"
            "Living in a large city and living in a rural area can offer very different "
            "advantages. A city often provides more public transportation, restaurants, "
            "schools, and job opportunities. A rural area, by contrast,"
        ),
    ),
    (
        "compare_books_movies",
        (
            "Comparison:\n"
            "Books and movies can tell the same story in different ways. A book can "
            "spend more time describing a character's thoughts, memories, and private "
            "reactions. A movie, on the other hand,"
        ),
    ),
    (
        "compare_online_in_person_learning",
        (
            "Comparison:\n"
            "Online learning and in-person learning each have strengths. Online courses "
            "can be flexible because students may watch lectures or complete assignments "
            "from different locations. In-person classes can be useful because"
        ),
    ),
    (
        "compare_solar_wind",
        (
            "Comparison:\n"
            "Solar power and wind power are both renewable energy sources, but they "
            "depend on different natural conditions. Solar panels produce electricity "
            "from sunlight, while wind turbines"
        ),
    ),

    # -------------------------------------------------------------------------
    # Structured data / semi-formal text
    # -------------------------------------------------------------------------
    (
        "meeting_notes_project",
        (
            "Meeting notes: Community garden planning\n\n"
            "Date: April 12\n"
            "Attendees: Lena, Marcus, Priya, Omar\n\n"
            "Topics discussed:\n"
            "1. The group reviewed possible locations for the garden.\n"
            "2. Priya suggested contacting the school about unused land near the playground.\n"
            "3."
        ),
    ),
    (
        "recipe_pancakes",
        (
            "Recipe: Simple banana pancakes\n\n"
            "Ingredients:\n"
            "- 2 ripe bananas\n"
            "- 2 eggs\n"
            "- 1/2 cup oats\n"
            "- 1/2 teaspoon cinnamon\n\n"
            "Instructions:\n"
            "1. Mash the bananas in a bowl until smooth.\n"
            "2."
        ),
    ),
    (
        "travel_itinerary",
        (
            "Weekend itinerary: Small coastal town\n\n"
            "Saturday morning:\n"
            "Arrive by train and walk from the station to the harbor. Stop at a cafe "
            "for breakfast and look over the map of the old town.\n\n"
            "Saturday afternoon:\n"
        ),
    ),
    (
        "product_review",
        (
            "Product review: Lightweight hiking backpack\n\n"
            "I used this backpack on several day hikes in mild spring weather. The first "
            "thing I noticed was that the shoulder straps were comfortable even after "
            "two hours of walking. The main compartment"
        ),
    ),

    # -------------------------------------------------------------------------
    # Low-entropy diagnostics: useful, but do not average them with everything
    # -------------------------------------------------------------------------
    (
        "diagnostic_counting",
        (
            "Sequence:\n"
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,"
        ),
    ),
    (
        "diagnostic_alphabet",
        (
            "Alphabet sequence:\n"
            "A, B, C, D, E, F, G,"
        ),
    ),
]


INFERENCE_TEST_PROMPTS_EASY = [
    (
        "easy_summary_garden",
        (
            "Passage:\n"
            "The community garden opened in May. Volunteers planted tomatoes, carrots, "
            "lettuce, and herbs. Local families visit the garden on weekends to water "
            "the plants and collect vegetables.\n\n"
            "Summary:\n"
        ),
    ),
    (
        "easy_summary_library",
        (
            "Passage:\n"
            "The town library added a new reading room with large windows and comfortable "
            "chairs. Students often use the room after school, and older residents visit "
            "in the morning to read newspapers.\n\n"
            "Summary:\n"
        ),
    ),
    (
        "easy_fact_animals",
        (
            "Article: Elephants\n\n"
            "Elephants are large mammals known for their long trunks, wide ears, and "
            "strong social bonds. They live in groups and use their trunks to"
        ),
    ),
    (
        "easy_fact_rain",
        (
            "Article: Rain\n\n"
            "Rain forms when water vapor in the atmosphere cools and condenses into "
            "droplets. When the droplets become heavy enough, they fall to the ground as"
        ),
    ),
    (
        "easy_qa_water",
        (
            "Question:\n"
            "Why do people need clean drinking water?\n\n"
            "Answer:\n"
        ),
    ),
    (
        "easy_qa_plants",
        (
            "Question:\n"
            "Why do plants need sunlight?\n\n"
            "Answer:\n"
        ),
    ),
    (
        "easy_math_cookies",
        (
            "Problem:\n"
            "Sara baked 12 cookies. She gave 4 cookies to her friend and ate 2 herself. "
            "How many cookies are left?\n\n"
            "Solution:\n"
        ),
    ),
    (
        "easy_math_books",
        (
            "Problem:\n"
            "There are 5 books on each shelf. The bookcase has 6 shelves. How many books "
            "are there in total?\n\n"
            "Solution:\n"
        ),
    ),
    (
        "easy_transform_formal",
        (
            "Original note:\n"
            "hi sam, thanks for helping me yesterday. i really appreciate it.\n\n"
            "Formal version:\n"
        ),
    ),
    (
        "easy_transform_paragraph",
        (
            "Notes:\n"
            "- The dog ran into the yard.\n"
            "- It chased a red ball.\n"
            "- The children laughed.\n\n"
            "Paragraph:\n"
        ),
    ),
    (
        "easy_code_add_numbers",
        (
            "def add_numbers(a: int, b: int) -> int:\n"
            "    \"\"\"Return the sum of two integers.\"\"\"\n"
        ),
    ),
    (
        "easy_code_is_even",
        (
            "def is_even(n: int) -> bool:\n"
            "    \"\"\"Return True if n is even, otherwise return False.\"\"\"\n"
        ),
    ),
    (
        "easy_story_cat",
        (
            "Short story opening:\n\n"
            "Milo the cat sat by the kitchen window and watched the rain fall outside. "
            "When the clouds began to clear,"
        ),
    ),
    (
        "easy_story_boat",
        (
            "Short story opening:\n\n"
            "The small wooden boat rested beside the quiet lake. Emma picked up the oars, "
            "stepped carefully inside, and"
        ),
    ),
    (
        "easy_procedure_tea",
        (
            "Guide: How to make a cup of tea\n\n"
            "First, boil fresh water in a kettle. Place a tea bag in a clean cup. When "
            "the water is ready,"
        ),
    ),
    (
        "easy_procedure_bed",
        (
            "Guide: How to make a bed\n\n"
            "Start by pulling the fitted sheet over the mattress. Smooth out the corners "
            "and place the pillows near the headboard. Then"
        ),
    ),
    (
        "easy_compare_cats_dogs",
        (
            "Comparison:\n"
            "Cats and dogs are both common household pets. Cats are often more independent, "
            "while dogs usually"
        ),
    ),
    (
        "easy_compare_bus_train",
        (
            "Comparison:\n"
            "Buses and trains both help people travel without driving a car. Buses can "
            "stop in many neighborhoods, while trains"
        ),
    ),
    (
        "easy_recipe_sandwich",
        (
            "Recipe: Simple cheese sandwich\n\n"
            "Ingredients:\n"
            "- 2 slices of bread\n"
            "- 1 slice of cheese\n"
            "- butter\n\n"
            "Instructions:\n"
            "1. Place the bread on a plate.\n"
            "2."
        ),
    ),
    (
        "easy_meeting_notes",
        (
            "Meeting notes: School picnic planning\n\n"
            "Topics discussed:\n"
            "1. The picnic will take place in the park.\n"
            "2. Parents will bring snacks and drinks.\n"
            "3."
        ),
    ),
]


INFERENCE_TEST_PROMPTS_CONCRETE = [
    # -------------------------------------------------------------------------
    # Concrete completion tasks with narrow, unambiguous outputs
    # -------------------------------------------------------------------------
    (
        "concrete_math_total_price",
        (
            "Task: Compute the total cost.\n"
            "A notebook costs 4 dollars. A pen costs 2 dollars. Buy 3 notebooks "
            "and 5 pens.\n"
            "Return only the total number of dollars.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_remaining_pages",
        (
            "Task: Compute the remaining pages.\n"
            "A book has 240 pages. Lina has read 85 pages on Monday and 70 pages "
            "on Tuesday.\n"
            "Return only the number of pages left unread.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_average",
        (
            "Task: Compute the arithmetic mean.\n"
            "Numbers: 6, 10, 14, 18\n"
            "Return only the mean as an integer.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_sort_numbers",
        (
            "Task: Sort the numbers in ascending order.\n"
            "Numbers: 42, 7, 19, 3, 28\n"
            "Return only a comma-separated list of numbers.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_find_largest",
        (
            "Task: Find the largest number.\n"
            "Numbers: 18, 73, 41, 9, 55\n"
            "Return only the largest number.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_count_words",
        (
            "Task: Count the words in the sentence.\n"
            "Sentence: The quiet train arrived before noon.\n"
            "Return only the word count as a number.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_extract_name",
        (
            "Task: Extract the customer's full name.\n"
            "Record: Customer: Maya Chen; Order: 3 notebooks; City: Boston.\n"
            "Return only the full name.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_extract_date",
        (
            "Task: Extract the event date.\n"
            "Text: The safety inspection is scheduled for March 14, 2027 at 09:00.\n"
            "Return only the date exactly as written.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_extract_email",
        (
            "Task: Extract the email address.\n"
            "Text: Please send the invoice to billing@example.com before Friday.\n"
            "Return only the email address.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_classify_sentiment",
        (
            "Task: Classify the sentiment of the review.\n"
            "Review: The headphones arrived on time and the sound quality is excellent.\n"
            "Choose exactly one label: positive, neutral, negative.\n\n"
            "Label:\n"
        ),
    ),
    (
        "concrete_classify_department",
        (
            "Task: Choose the correct support department.\n"
            "Message: I was charged twice for the same subscription renewal.\n"
            "Choose exactly one department: billing, technical, sales.\n\n"
            "Department:\n"
        ),
    ),
    (
        "concrete_yes_no",
        (
            "Task: Answer the question with yes or no.\n"
            "Rule: A package is late if it arrives after May 10.\n"
            "Package arrival date: May 12.\n"
            "Question: Is the package late?\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_unit_conversion",
        (
            "Task: Convert kilometers to meters.\n"
            "Distance: 3.5 kilometers\n"
            "Return only the number of meters.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_temperature_conversion",
        (
            "Task: Convert Celsius to Fahrenheit.\n"
            "Formula: F = C * 9 / 5 + 32\n"
            "Celsius: 20\n"
            "Return only the Fahrenheit value as an integer.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_format_initials",
        (
            "Task: Write the initials for the name.\n"
            "Name: Sofia Maria Alvarez\n"
            "Return only the initials with periods and no spaces.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_lowercase",
        (
            "Task: Convert the text to lowercase.\n"
            "Text: SMALL STEPS BUILD STRONG HABITS\n"
            "Return only the lowercase text.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_replace_word",
        (
            "Task: Replace every occurrence of blue with green.\n"
            "Text: The blue cup is beside the blue plate.\n"
            "Return only the updated sentence.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_json_object",
        (
            "Task: Create a JSON object from the fields.\n"
            "Name: Omar Patel\n"
            "Role: Designer\n"
            "Active: true\n"
            "Use keys exactly: name, role, active.\n"
            "Return only valid JSON on one line.\n\n"
            "JSON:\n"
        ),
    ),
    (
        "concrete_csv_row",
        (
            "Task: Create one CSV row.\n"
            "Fields in order: product, quantity, price\n"
            "Values: pencil, 12, 1.50\n"
            "Return only the CSV row with commas and no header.\n\n"
            "CSV:\n"
        ),
    ),
    (
        "concrete_code_completion",
        (
            "def multiply_by_three(n: int) -> int:\n"
            "    \"\"\"Return n multiplied by exactly 3.\"\"\"\n"
        ),
    ),
]


CHAT_TEST_PROMPTS = [
    # Factual / encyclopedic
    ("fact_photosynthesis",      "How does photosynthesis work?"),
    ("fact_moon",                "Why does the Moon have craters on its surface?"),
    ("fact_printing_press",      "How did the printing press change society in Europe?"),
    ("fact_volcanoes",           "Why do volcanoes often form near tectonic plate boundaries?"),
    ("fact_water_cycle",         "Can you explain the main stages of the water cycle?"),

    # Reasoning / explanation
    ("reason_supply_demand",     "How do supply and demand affect the price of a product?"),
    ("reason_democracy",         "Why do democratic systems hold regular elections?"),
    ("reason_greenhouse",        "Why does the greenhouse effect cause the Earth to warm up?"),
    ("reason_antibiotics",       "Why is it important to finish a full course of antibiotics?"),

    # Math word problems
    ("math_apples",              "Maya has 18 apples. She gives 5 to her neighbor and then buys 12 more. How many does she have now?"),
    ("math_tickets",             "A train ticket costs 7 dollars. A family buys 4 tickets and also pays 6 dollars for parking. What is the total cost?"),
    ("math_recipe",              "A recipe needs 3 cups of flour per cake. Lena wants to bake 5 cakes but already has 4 cups at home. How many more cups does she need?"),

    # Code generation
    ("code_reverse_words",       "Write a Python function that takes a string and returns it with the words in reverse order."),
    ("code_count_vowels",        "Write a Python function that counts the number of vowels in a string, ignoring case."),
    ("code_merge_dicts",         "Write a Python function that merges two dictionaries of integer counts, summing values for shared keys."),
    ("code_palindrome",          "Write a Python function that returns True if a string is a palindrome, ignoring spaces and case."),

    # Summarization
    ("summary_renewable",        "Summarize this in one or two sentences: Many cities are investing in solar panels, wind power, and LED streetlights to cut fossil fuel use. Officials say upfront costs will pay off through lower energy bills and cleaner air."),
    ("summary_school_lunch",     "Summarize this briefly: A school district redesigned its lunch menu after parent and teacher complaints about nutrition, adding more fresh produce and whole grains while reducing processed food."),

    # Transformation / rewriting
    ("transform_formal",         "Rewrite this message in a formal tone: hey, i can't make it to the meeting today. can we move it to tomorrow?"),
    ("transform_active",         "Rewrite this sentence in active voice: The report was reviewed by the committee before the announcement was made."),
    ("transform_simple",         "Explain this in simple terms for a ten-year-old: Evaporation occurs when molecules at the surface of a liquid gain enough energy to enter the gas phase."),

    # Comparison
    ("compare_city_rural",       "What are the main differences between living in a big city and living in the countryside?"),
    ("compare_solar_wind",       "What are the differences between solar power and wind power as energy sources?"),
    ("compare_books_movies",     "How does reading a book differ from watching a movie adaptation of the same story?"),

    # Creative / open-ended
    ("story_lighthouse",         "Write the opening paragraph of a short story about a lighthouse keeper who discovers something unusual in the fog."),
    ("story_robot",              "Write the opening paragraph of a short story about a small garden robot that finds a mysterious seed."),

    # Procedural
    ("procedure_seedlings",      "What are the basic steps for starting vegetable seedlings indoors?"),
    ("procedure_flat_tire",      "How do you fix a flat bicycle tire?"),
]

# INFERENCE_TEST_PROMPTS = [
#     (
#         "Recent U.S. presidents list",
#         (
#             "The 10 most recent U.S. presidents, listed once each in reverse chronological order, are:\n"
#             "1. Donald Trump\n"
#             "2."
#         ),
#     ),
#     (
#         "Talking about Paris",
#         (
#             "Short factual paragraph, no repeated sentences:\n"
#             "Paris is the capital of France. It is"
#         ),
#     ),
#     (
#         "Counting to 100",
#         (
#             "Count from 1 to 100, without restarting or repeating any number:\n"
#             "1, 2, 3, 4,"
#         ),
#     ),
#     (
#         "Story about Bob",
#         (
#             "A short five-sentence story with a clear ending:\n"
#             "Bob lived in Texas and liked to drive his pickup truck."
#         ),
#     ),
#     (
#         "Story about Bob, alternate",
#         (
#             "One paragraph story. Each sentence adds a new event. No repeated phrases:\n"
#             "Bob lived in Texas and liked to drive his pickup truck."
#         ),
#     ),
# ]
# # [
# #     (
# #         "Recent U.S. presidents list",
# #         "What are the 10 last presidents of the USA?",
# #     ),
# #     (
# #         "Talking about Paris",
# #         "What is the capital of France?",
# #     ),
# #      (
# #         "Counting to 100",
# #         "Count to 100, like 1, 2, 3, 4, etc.",
# #     ),
# #     (
# #         "Story about Bob",
# #         "Tell me a story about a man named Bob that lived in Texas and liked to drive his pickup truck."
# #     ),
# # ]
