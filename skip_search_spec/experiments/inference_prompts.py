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
    (
        "concrete_math_invoice_total",
        (
            "Task: Compute the invoice total after discount.\n"
            "Items: 4 lamps at 18 dollars each, 2 chairs at 45 dollars each. "
            "Apply a 10 dollar discount to the combined price.\n"
            "Return only the final number of dollars.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_split_bill",
        (
            "Task: Compute each person's share.\n"
            "A restaurant bill is 96 dollars. Add a 20 percent tip, then split the "
            "total equally among 4 people.\n"
            "Return only the number of dollars per person.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_percent_remaining",
        (
            "Task: Compute the percentage remaining.\n"
            "A battery has 80 watt-hours when full. It has used 26 watt-hours.\n"
            "Return only the remaining percentage as a number.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_weighted_score",
        (
            "Task: Compute the weighted score.\n"
            "Homework is worth 40 percent and exam is worth 60 percent. "
            "Homework score: 85. Exam score: 70.\n"
            "Return only the final weighted score as a number.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_inventory_after_sales",
        (
            "Task: Compute the final inventory.\n"
            "Start with 125 mugs. Receive 48 more mugs. Sell 37 mugs in the morning "
            "and 29 mugs in the afternoon.\n"
            "Return only the number of mugs left.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_recipe_scaling",
        (
            "Task: Scale the recipe ingredient.\n"
            "A recipe for 6 servings uses 450 grams of rice. Scale it to 10 servings.\n"
            "Return only the number of grams of rice.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_speed_distance",
        (
            "Task: Compute the travel distance.\n"
            "A cyclist rides at 18 kilometers per hour for 2.5 hours.\n"
            "Return only the number of kilometers traveled.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_train_arrival",
        (
            "Task: Compute the arrival time.\n"
            "A train leaves at 14:35. The trip takes 2 hours and 48 minutes.\n"
            "Return only the arrival time in 24-hour HH:MM format.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_meeting_duration",
        (
            "Task: Compute the meeting duration.\n"
            "A meeting starts at 09:15 and ends at 11:05.\n"
            "Return only the duration in minutes.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_monthly_savings",
        (
            "Task: Compute the number of months needed.\n"
            "Target savings: 900 dollars. Already saved: 180 dollars. Saves 120 "
            "dollars per month.\n"
            "Return only the number of full months needed to reach the target.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_box_volume",
        (
            "Task: Compute the box volume.\n"
            "A box is 12 cm long, 8 cm wide, and 5 cm tall.\n"
            "Return only the volume in cubic centimeters.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_rectangle_perimeter",
        (
            "Task: Compute the perimeter.\n"
            "A rectangle has length 17 meters and width 9 meters.\n"
            "Return only the perimeter in meters.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_average_with_new_value",
        (
            "Task: Compute the new average.\n"
            "The current average of 5 tests is 82. A sixth test score is 94.\n"
            "Return only the new average as a number.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_unit_price",
        (
            "Task: Compute the unit price.\n"
            "A 750 gram bag of coffee costs 12 dollars.\n"
            "Return only the price per kilogram in dollars.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_math_change_due",
        (
            "Task: Compute the change due.\n"
            "A customer buys items costing 13.75 dollars and pays with a 20 dollar bill.\n"
            "Return only the change in dollars with two decimal places.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_sort_dates",
        (
            "Task: Sort the dates from earliest to latest.\n"
            "Dates: 2026-03-12, 2025-11-04, 2026-01-20, 2025-12-31\n"
            "Return only a comma-separated list of dates.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_sort_names_by_last_name",
        (
            "Task: Sort the full names alphabetically by last name.\n"
            "Names: Priya Nair, Lucas Berg, Emma Chen, Noah Alvarez\n"
            "Return only the names as a comma-separated list.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_filter_even_numbers",
        (
            "Task: Keep only the even numbers, preserving the original order.\n"
            "Numbers: 15, 22, 8, 31, 44, 57, 60\n"
            "Return only a comma-separated list of numbers.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_filter_over_threshold",
        (
            "Task: Keep only values greater than 50, preserving the original order.\n"
            "Values: 48, 72, 50, 91, 13, 65\n"
            "Return only a comma-separated list of values.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_find_second_largest",
        (
            "Task: Find the second largest number.\n"
            "Numbers: 14, 89, 52, 89, 73, 41\n"
            "Treat repeated values as one value. Return only the second largest number.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_count_unique_words",
        (
            "Task: Count the unique words, ignoring case.\n"
            "Sentence: Red fish blue fish red bird.\n"
            "Return only the number of unique words.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_count_letter_occurrences",
        (
            "Task: Count how many times the letter a appears, ignoring case.\n"
            "Text: A calm canal ran past Anita's garden.\n"
            "Return only the count as a number.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_find_missing_number",
        (
            "Task: Find the missing number in the sequence.\n"
            "Sequence: 4, 8, 12, 16, __, 24\n"
            "Return only the missing number.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_find_duplicate_id",
        (
            "Task: Find the duplicate ID.\n"
            "IDs: T104, T211, T305, T104, T480\n"
            "Return only the duplicate ID.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_extract_phone_number",
        (
            "Task: Extract the phone number.\n"
            "Text: For urgent delivery questions, call +1-415-555-0198 after 08:00.\n"
            "Return only the phone number.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_extract_postal_code",
        (
            "Task: Extract the postal code.\n"
            "Address: 18 Cedar Lane, Portland, OR 97205, USA.\n"
            "Return only the postal code.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_extract_order_id",
        (
            "Task: Extract the order ID.\n"
            "Message: Shipment for order ORD-7842-A has been delayed by one day.\n"
            "Return only the order ID.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_extract_currency_amount",
        (
            "Task: Extract the refund amount.\n"
            "Text: The customer will receive a refund of $47.90 within five business days.\n"
            "Return only the amount including the currency symbol.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_extract_city",
        (
            "Task: Extract the destination city.\n"
            "Ticket: Passenger: Nora Blake; From: Denver; To: Seattle; Seat: 12C.\n"
            "Return only the destination city.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_extract_after_marker",
        (
            "Task: Extract the value after Status.\n"
            "Record: Ticket=PX-44; Owner=Ravi Singh; Status=waiting-review; Priority=high.\n"
            "Return only the status value.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_extract_between_tags",
        (
            "Task: Extract the text inside the title tags.\n"
            "Text: <note><title>Quarterly Budget Review</title><owner>Finance</owner></note>\n"
            "Return only the title text.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_extract_filename_extension",
        (
            "Task: Extract the file extension.\n"
            "Filename: archive.final.v3.tar.gz\n"
            "Return only the extension after the last period.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_extract_domain",
        (
            "Task: Extract the domain from the URL.\n"
            "URL: https://docs.example.org/projects/alpha?tab=files\n"
            "Return only the domain.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_extract_hashtag",
        (
            "Task: Extract the hashtag.\n"
            "Post: Launch photos are live now #NorthPierOpening thanks to the whole team.\n"
            "Return only the hashtag including the # symbol.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_format_iso_date",
        (
            "Task: Convert the date to ISO format.\n"
            "Date: July 9, 2026\n"
            "Return only the date in YYYY-MM-DD format.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_format_us_date",
        (
            "Task: Convert the date to MM/DD/YYYY format.\n"
            "Date: 2027-02-03\n"
            "Return only the formatted date.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_format_phone_digits",
        (
            "Task: Remove all non-digit characters from the phone number.\n"
            "Phone: (212) 555-0147 ext. 9\n"
            "Return only the digits.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_format_title_case",
        (
            "Task: Convert the text to title case.\n"
            "Text: annual report for northern harbor project\n"
            "Return only the title-cased text.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_format_slug",
        (
            "Task: Convert the title to a URL slug.\n"
            "Title: Winter Market Schedule Update\n"
            "Use lowercase words separated by hyphens. Return only the slug.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_format_snake_case",
        (
            "Task: Convert the label to snake_case.\n"
            "Label: Customer Renewal Date\n"
            "Return only the snake_case label.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_format_currency",
        (
            "Task: Format the amount as US currency.\n"
            "Amount: 1287.5\n"
            "Return only the formatted amount with a dollar sign and two decimals.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_format_fixed_decimals",
        (
            "Task: Round the number to two decimal places.\n"
            "Number: 19.8764\n"
            "Return only the rounded number with exactly two decimals.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_format_join_names",
        (
            "Task: Join the names with semicolons.\n"
            "Names: Aria, Ben, Carmen, Diego\n"
            "Return only the semicolon-separated string.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_format_reverse_words",
        (
            "Task: Reverse the word order.\n"
            "Text: clear plans reduce project risk\n"
            "Return only the words in reverse order.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_transform_remove_vowels",
        (
            "Task: Remove all vowels from the text.\n"
            "Text: reliable backup system\n"
            "Treat a, e, i, o, u as vowels. Return only the transformed text.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_transform_abbreviate_months",
        (
            "Task: Replace full month names with three-letter abbreviations.\n"
            "Text: Reports are due in January, April, and September.\n"
            "Return only the updated sentence.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_transform_mask_email",
        (
            "Task: Mask the email username after the first two characters.\n"
            "Email: jonathan.rivera@example.com\n"
            "Keep the domain unchanged. Return only the masked email.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_classify_priority",
        (
            "Task: Classify the support ticket priority.\n"
            "Ticket: The production payment service is unavailable for all customers.\n"
            "Choose exactly one label: low, medium, high, critical.\n\n"
            "Priority:\n"
        ),
    ),
    (
        "concrete_classify_file_type",
        (
            "Task: Classify the file type from the extension.\n"
            "Filename: presentation.final.pptx\n"
            "Choose exactly one label: document, spreadsheet, presentation, image.\n\n"
            "Type:\n"
        ),
    ),
    (
        "concrete_classify_weather_advice",
        (
            "Task: Choose the most appropriate advice.\n"
            "Forecast: Heavy rain is expected from 15:00 to 19:00.\n"
            "Choose exactly one option: bring umbrella, wear sunglasses, water plants.\n\n"
            "Advice:\n"
        ),
    ),
    (
        "concrete_classify_language",
        (
            "Task: Identify the language.\n"
            "Sentence: El tren llega a la estacion central a las ocho.\n"
            "Choose exactly one label: English, Spanish, French, German.\n\n"
            "Language:\n"
        ),
    ),
    (
        "concrete_classify_topic",
        (
            "Task: Classify the topic of the sentence.\n"
            "Sentence: The committee approved the annual budget after reviewing tax revenue.\n"
            "Choose exactly one topic: finance, sports, cooking, travel.\n\n"
            "Topic:\n"
        ),
    ),
    (
        "concrete_classify_shipping_status",
        (
            "Task: Classify the shipping status.\n"
            "Message: The package left the regional sorting center at 06:40 and is on the truck.\n"
            "Choose exactly one status: pending, in_transit, delivered, cancelled.\n\n"
            "Status:\n"
        ),
    ),
    (
        "concrete_classify_access_level",
        (
            "Task: Choose the access level.\n"
            "Rule: Interns can view reports. Managers can view and edit reports. "
            "Admins can view, edit, and delete reports.\n"
            "User role: Manager\n"
            "Choose exactly one level: view_only, edit_allowed, delete_allowed.\n\n"
            "Level:\n"
        ),
    ),
    (
        "concrete_classify_pass_fail",
        (
            "Task: Determine whether the student passed.\n"
            "Rule: Passing requires a score of at least 60.\n"
            "Score: 58\n"
            "Choose exactly one label: pass, fail.\n\n"
            "Label:\n"
        ),
    ),
    (
        "concrete_classify_age_group",
        (
            "Task: Classify the age group.\n"
            "Rule: child is under 13, teen is 13 to 17, adult is 18 or older.\n"
            "Age: 17\n"
            "Choose exactly one label: child, teen, adult.\n\n"
            "Group:\n"
        ),
    ),
    (
        "concrete_classify_inventory_status",
        (
            "Task: Classify the inventory status.\n"
            "Rule: 0 units is out_of_stock, 1-20 units is low_stock, above 20 is in_stock.\n"
            "Units available: 20\n"
            "Choose exactly one label: out_of_stock, low_stock, in_stock.\n\n"
            "Status:\n"
        ),
    ),
    (
        "concrete_boolean_weekend",
        (
            "Task: Answer with yes or no.\n"
            "Question: Is Saturday a weekend day?\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_boolean_threshold",
        (
            "Task: Answer with yes or no.\n"
            "Rule: Free shipping applies when the order total is at least 75 dollars.\n"
            "Order total: 74.99 dollars.\n"
            "Question: Does free shipping apply?\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_boolean_date_range",
        (
            "Task: Answer with yes or no.\n"
            "Allowed booking dates: 2026-06-01 through 2026-06-30 inclusive.\n"
            "Requested date: 2026-06-30.\n"
            "Question: Is the requested date allowed?\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_boolean_contains_word",
        (
            "Task: Answer with yes or no.\n"
            "Text: The archive includes invoices, receipts, and contracts.\n"
            "Question: Does the text contain the word receipts?\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_boolean_all_conditions",
        (
            "Task: Answer with yes or no.\n"
            "Rule: A loan is approved only if income is at least 3000 and credit score "
            "is at least 680.\n"
            "Income: 3200. Credit score: 675.\n"
            "Question: Is the loan approved?\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_boolean_password_valid",
        (
            "Task: Answer with yes or no.\n"
            "Rule: A password is valid if it has at least 8 characters and contains a digit.\n"
            "Password: meadow42\n"
            "Question: Is the password valid?\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_date_next_day",
        (
            "Task: Compute the next calendar day.\n"
            "Date: 2026-02-28\n"
            "Return only the next date in YYYY-MM-DD format.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_date_previous_day",
        (
            "Task: Compute the previous calendar day.\n"
            "Date: 2026-03-01\n"
            "Return only the previous date in YYYY-MM-DD format.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_date_add_days",
        (
            "Task: Add days to the date.\n"
            "Start date: 2026-05-17. Add 45 days.\n"
            "Return only the resulting date in YYYY-MM-DD format.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_date_weekday_from_schedule",
        (
            "Task: Find the weekday for the appointment.\n"
            "Schedule: Monday 09:00 dentist, Tuesday 14:00 library, Wednesday 11:30 bank.\n"
            "Appointment: library\n"
            "Return only the weekday.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_time_timezone_simple",
        (
            "Task: Convert the time from UTC to UTC+2.\n"
            "UTC time: 18:45\n"
            "Return only the local time in HH:MM format.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_time_duration_across_midnight",
        (
            "Task: Compute the elapsed time.\n"
            "Start: 22:50. End: 01:15 the next day.\n"
            "Return only the elapsed time in minutes.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_table_lookup_price",
        (
            "Task: Look up the product price.\n"
            "Table:\n"
            "Product | Price\n"
            "Notebook | 4.50\n"
            "Marker | 1.25\n"
            "Folder | 2.10\n"
            "Question: What is the price of Marker?\n"
            "Return only the price.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_table_lookup_manager",
        (
            "Task: Look up the team manager.\n"
            "Table:\n"
            "Team | Manager\n"
            "Design | Elena\n"
            "Platform | Omar\n"
            "Support | Nina\n"
            "Question: Who manages Platform?\n"
            "Return only the manager name.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_table_sum_column",
        (
            "Task: Sum the quantity column.\n"
            "Rows:\n"
            "apples, 14\n"
            "oranges, 9\n"
            "pears, 17\n"
            "Return only the total quantity.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_table_max_row",
        (
            "Task: Find the city with the highest population.\n"
            "Rows:\n"
            "Arden, 82000\n"
            "Briggs, 91500\n"
            "Claremont, 77600\n"
            "Return only the city name.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_table_filter_status",
        (
            "Task: List the ticket IDs with status open, preserving table order.\n"
            "Rows:\n"
            "A17, open\n"
            "B22, closed\n"
            "C09, open\n"
            "D14, waiting\n"
            "Return only a comma-separated list of IDs.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_table_average_score",
        (
            "Task: Compute the average score.\n"
            "Rows:\n"
            "Mila, 88\n"
            "Jon, 76\n"
            "Tara, 91\n"
            "Return only the average score as a number.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_json_nested_value",
        (
            "Task: Extract a value from the JSON.\n"
            "JSON: {\"user\":{\"name\":\"Iris Lee\",\"role\":\"editor\"},\"active\":true}\n"
            "Return only the user's role.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_json_array_count",
        (
            "Task: Count the items in the JSON array.\n"
            "JSON: {\"tags\":[\"urgent\",\"finance\",\"review\",\"q2\"]}\n"
            "Return only the number of tags.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_json_boolean_value",
        (
            "Task: Extract the boolean value from the JSON.\n"
            "JSON: {\"feature\":\"exports\",\"enabled\":false,\"owner\":\"ops\"}\n"
            "Return only the value of enabled.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_json_create_array",
        (
            "Task: Create a JSON array from the values.\n"
            "Values: red, blue, green\n"
            "Return only valid JSON on one line.\n\n"
            "JSON:\n"
        ),
    ),
    (
        "concrete_json_create_object_typed",
        (
            "Task: Create a JSON object from the fields.\n"
            "ID: 42\n"
            "Name: North Gate\n"
            "Open: false\n"
            "Use keys exactly: id, name, open. Return only valid JSON on one line.\n\n"
            "JSON:\n"
        ),
    ),
    (
        "concrete_json_update_field",
        (
            "Task: Update one field in the JSON object.\n"
            "JSON: {\"name\":\"Basic Plan\",\"price\":12,\"active\":true}\n"
            "Change price to 15. Return only the updated JSON on one line.\n\n"
            "JSON:\n"
        ),
    ),
    (
        "concrete_csv_extract_second_field",
        (
            "Task: Extract the second field from the CSV row.\n"
            "CSV row: Rivera,42,active,Seattle\n"
            "Return only the second field.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_csv_create_header",
        (
            "Task: Create a CSV header.\n"
            "Fields in order: date, customer, total, paid\n"
            "Return only the CSV header row.\n\n"
            "CSV:\n"
        ),
    ),
    (
        "concrete_csv_quote_field",
        (
            "Task: Create one CSV row, quoting fields when needed.\n"
            "Fields in order: name, note, amount\n"
            "Values: Samir Khan; paid, waiting for receipt; 31.20\n"
            "Return only the CSV row with no header.\n\n"
            "CSV:\n"
        ),
    ),
    (
        "concrete_csv_count_rows",
        (
            "Task: Count the data rows, excluding the header.\n"
            "CSV:\n"
            "name,total\n"
            "Ada,14\n"
            "Bo,27\n"
            "Cy,19\n"
            "Return only the number of data rows.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_list_intersection",
        (
            "Task: Find the values that appear in both lists, preserving the first list order.\n"
            "List A: red, green, blue, yellow\n"
            "List B: black, yellow, red, white\n"
            "Return only a comma-separated list.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_list_difference",
        (
            "Task: Find the values in List A that are not in List B, preserving order.\n"
            "List A: oak, pine, birch, maple\n"
            "List B: maple, cedar, pine\n"
            "Return only a comma-separated list.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_list_deduplicate",
        (
            "Task: Remove duplicates while preserving the first occurrence.\n"
            "Items: tea, coffee, tea, juice, coffee, water\n"
            "Return only a comma-separated list.\n\n"
            "Output:\n"
        ),
    ),
    (
        "concrete_list_group_count",
        (
            "Task: Count how many items belong to the fruit category.\n"
            "Items: apple=fruit, carrot=vegetable, pear=fruit, onion=vegetable, plum=fruit\n"
            "Return only the fruit count.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_logic_seating_left_of",
        (
            "Task: Determine who sits immediately left of Omar.\n"
            "Seats from left to right: Lina, Omar, Chen, Fatima.\n"
            "Return only the person's name.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_logic_rank_order",
        (
            "Task: Determine who finished second.\n"
            "Race results: Mei finished before Noah. Noah finished before Iris. "
            "Luca finished after Iris.\n"
            "Return only the second-place name.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_logic_schedule_conflict",
        (
            "Task: Identify the conflicting meeting.\n"
            "Existing meeting: 10:00-11:00.\n"
            "Candidates: A 09:00-09:30, B 10:30-11:30, C 11:15-11:45.\n"
            "Return only the candidate letter that overlaps the existing meeting.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_logic_route_next_stop",
        (
            "Task: Find the next stop after Central.\n"
            "Route: Harbor -> Museum -> Central -> Garden -> Airport\n"
            "Return only the next stop name.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_logic_parent_child",
        (
            "Task: Identify the grandparent.\n"
            "Facts: Elena is Marco's parent. Marco is Sofia's parent.\n"
            "Question: Who is Sofia's grandparent?\n"
            "Return only the name.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_logic_grid_cell",
        (
            "Task: Find the value at row 2, column 3.\n"
            "Grid:\n"
            "Row 1: A B C\n"
            "Row 2: D E F\n"
            "Row 3: G H I\n"
            "Return only the cell value.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_unit_pounds_to_ounces",
        (
            "Task: Convert pounds to ounces.\n"
            "Weight: 6.5 pounds\n"
            "Use 16 ounces per pound. Return only the number of ounces.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_unit_minutes_to_hours",
        (
            "Task: Convert minutes to hours and minutes.\n"
            "Duration: 145 minutes\n"
            "Return only the duration in the format H hours M minutes.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_unit_liters_to_milliliters",
        (
            "Task: Convert liters to milliliters.\n"
            "Volume: 2.75 liters\n"
            "Return only the number of milliliters.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_unit_inches_to_feet",
        (
            "Task: Convert inches to feet and inches.\n"
            "Length: 74 inches\n"
            "Return only the length in the format F feet I inches.\n\n"
            "Answer:\n"
        ),
    ),
    (
        "concrete_code_complete_add_tax",
        (
            "def add_tax(price: float) -> float:\n"
            "    \"\"\"Return price after adding exactly 8 percent tax.\"\"\"\n"
        ),
    ),
]


INFERENCE_TEST_PROMPTS_CONCRETE_SWEDISH = [
    # -------------------------------------------------------------------------
    # Konkreta svenska completion-uppgifter med smala, entydiga svar
    # -------------------------------------------------------------------------
    (
        "svenska_konkret_matte_totalpris",
        (
            "Uppgift: Räkna ut totalpriset.\n"
            "En anteckningsbok kostar 40 kronor. En penna kostar 15 kronor. "
            "Köp 2 anteckningsböcker och 3 pennor.\n"
            "Svara endast med det totala antalet kronor.\n\n"
            "Svar:\n"
        ),
    ),
    (
        "svenska_konkret_matte_sidor_kvar",
        (
            "Uppgift: Räkna ut hur många sidor som är kvar.\n"
            "En bok har 180 sidor. Sara läser 45 sidor på måndagen och 60 sidor "
            "på tisdagen.\n"
            "Svara endast med antalet olästa sidor.\n\n"
            "Svar:\n"
        ),
    ),
    (
        "svenska_konkret_matte_medelvarde",
        (
            "Uppgift: Räkna ut medelvärdet.\n"
            "Tal: 8, 12, 16, 20\n"
            "Svara endast med medelvärdet som ett heltal.\n\n"
            "Svar:\n"
        ),
    ),
    (
        "svenska_konkret_sortera_tal",
        (
            "Uppgift: Sortera talen i stigande ordning.\n"
            "Tal: 31, 4, 18, 9, 25\n"
            "Svara endast med en kommaseparerad lista av tal.\n\n"
            "Utdata:\n"
        ),
    ),
    (
        "svenska_konkret_storsta_tal",
        (
            "Uppgift: Hitta det största talet.\n"
            "Tal: 16, 72, 48, 91, 33\n"
            "Svara endast med det största talet.\n\n"
            "Svar:\n"
        ),
    ),
    (
        "svenska_konkret_rakna_ord",
        (
            "Uppgift: Räkna orden i meningen.\n"
            "Mening: Den röda bussen stannar vid torget.\n"
            "Svara endast med antalet ord som en siffra.\n\n"
            "Svar:\n"
        ),
    ),
    (
        "svenska_konkret_extrahera_namn",
        (
            "Uppgift: Extrahera kundens fullständiga namn.\n"
            "Post: Kund: Erik Lund; Beställning: 4 mappar; Stad: Uppsala.\n"
            "Svara endast med det fullständiga namnet.\n\n"
            "Utdata:\n"
        ),
    ),
    (
        "svenska_konkret_extrahera_datum",
        (
            "Uppgift: Extrahera datumet för mötet.\n"
            "Text: Projektmötet hålls den 18 september 2027 klockan 13:30.\n"
            "Svara endast med datumet exakt som det står i texten.\n\n"
            "Utdata:\n"
        ),
    ),
    (
        "svenska_konkret_extrahera_epost",
        (
            "Uppgift: Extrahera e-postadressen.\n"
            "Text: Skicka kvittot till ekonomi@example.se senast på fredag.\n"
            "Svara endast med e-postadressen.\n\n"
            "Utdata:\n"
        ),
    ),
    (
        "svenska_konkret_klassificera_sentiment",
        (
            "Uppgift: Klassificera känslan i recensionen.\n"
            "Recension: Leveransen kom snabbt och stolen var bekväm.\n"
            "Välj exakt en etikett: positiv, neutral, negativ.\n\n"
            "Etikett:\n"
        ),
    ),
    (
        "svenska_konkret_klassificera_avdelning",
        (
            "Uppgift: Välj rätt supportavdelning.\n"
            "Meddelande: Jag kan inte logga in på mitt konto efter lösenordsbytet.\n"
            "Välj exakt en avdelning: fakturering, teknik, försäljning.\n\n"
            "Avdelning:\n"
        ),
    ),
    (
        "svenska_konkret_ja_nej",
        (
            "Uppgift: Besvara frågan med ja eller nej.\n"
            "Regel: En faktura är sen om den betalas efter den 10 juni.\n"
            "Betalningsdatum: 12 juni.\n"
            "Fråga: Är fakturan sen?\n\n"
            "Svar:\n"
        ),
    ),
    (
        "svenska_konkret_enhetsomvandling",
        (
            "Uppgift: Omvandla kilometer till meter.\n"
            "Sträcka: 2,5 kilometer\n"
            "Svara endast med antalet meter.\n\n"
            "Svar:\n"
        ),
    ),
    (
        "svenska_konkret_initialer",
        (
            "Uppgift: Skriv initialerna för namnet.\n"
            "Namn: Anna Karin Berg\n"
            "Svara endast med initialerna med punkter och utan mellanslag.\n\n"
            "Utdata:\n"
        ),
    ),
    (
        "svenska_konkret_gemener",
        (
            "Uppgift: Gör texten till gemener.\n"
            "Text: SMÅ STEG BYGGER STARKA VANOR\n"
            "Svara endast med texten i gemener.\n\n"
            "Utdata:\n"
        ),
    ),
    (
        "svenska_konkret_ersatt_ord",
        (
            "Uppgift: Ersätt varje förekomst av blå med grön.\n"
            "Text: Den blå koppen står bredvid den blå tallriken.\n"
            "Svara endast med den uppdaterade meningen.\n\n"
            "Utdata:\n"
        ),
    ),
    (
        "svenska_konkret_json_objekt",
        (
            "Uppgift: Skapa ett JSON-objekt från fälten.\n"
            "Namn: Linnea Holm\n"
            "Roll: Analytiker\n"
            "Aktiv: true\n"
            "Använd exakt nycklarna: namn, roll, aktiv.\n"
            "Svara endast med giltig JSON på en rad.\n\n"
            "JSON:\n"
        ),
    ),
    (
        "svenska_konkret_csv_rad",
        (
            "Uppgift: Skapa en CSV-rad.\n"
            "Fält i ordning: produkt, antal, pris\n"
            "Värden: suddgummi, 6, 12.50\n"
            "Svara endast med CSV-raden med kommatecken och utan rubrik.\n\n"
            "CSV:\n"
        ),
    ),
    (
        "svenska_konkret_kod_completion",
        (
            "def dubbla_talet(n: int) -> int:\n"
            "    \"\"\"Returnera n multiplicerat med exakt 2.\"\"\"\n"
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
