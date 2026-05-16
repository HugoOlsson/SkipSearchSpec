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


INFERENCE_TEST_PROMPTS_PYTHON_COMPLETION = [
    (
        "python_complete_add_numbers",
        (
            "def add_numbers(a: int, b: int) -> int:\n"
            "    \"\"\"Return the sum of two integers.\n"
            "    \n"
            "    Example:\n"
            "    add_numbers(4, 7) == 11\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_clamp",
        (
            "def clamp(value: int, lower: int, upper: int) -> int:\n"
            "    \"\"\"Return value limited to the inclusive range [lower, upper].\n"
            "    \n"
            "    Example:\n"
            "    clamp(12, 0, 10) == 10\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_is_even",
        (
            "def is_even(number: int) -> bool:\n"
            "    \"\"\"Return True when number is divisible by 2.\n"
            "    \n"
            "    Example:\n"
            "    is_even(18) == True\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_factorial",
        (
            "def factorial(n: int) -> int:\n"
            "    \"\"\"Return n factorial for a non-negative integer n.\n"
            "    \n"
            "    Example:\n"
            "    factorial(5) == 120\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_fibonacci",
        (
            "def fibonacci(n: int) -> int:\n"
            "    \"\"\"Return the nth Fibonacci number using zero-based indexing.\n"
            "    \n"
            "    Examples:\n"
            "    fibonacci(0) == 0\n"
            "    fibonacci(6) == 8\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_gcd",
        (
            "def gcd(a: int, b: int) -> int:\n"
            "    \"\"\"Return the greatest common divisor of a and b.\n"
            "    \n"
            "    Example:\n"
            "    gcd(54, 24) == 6\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_lcm",
        (
            "def lcm(a: int, b: int) -> int:\n"
            "    \"\"\"Return the least common multiple of a and b.\n"
            "    \n"
            "    Example:\n"
            "    lcm(6, 8) == 24\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_is_prime",
        (
            "def is_prime(n: int) -> bool:\n"
            "    \"\"\"Return True if n is a prime number.\n"
            "    \n"
            "    Example:\n"
            "    is_prime(29) == True\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_sum_of_digits",
        (
            "def sum_of_digits(n: int) -> int:\n"
            "    \"\"\"Return the sum of the decimal digits of n.\n"
            "    \n"
            "    The sign of n should be ignored.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_round_to_places",
        (
            "def round_to_places(value: float, places: int) -> float:\n"
            "    \"\"\"Return value rounded to the requested number of decimal places.\n"
            "    \n"
            "    Example:\n"
            "    round_to_places(3.14159, 2) == 3.14\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_normalize_whitespace",
        (
            "def normalize_whitespace(text: str) -> str:\n"
            "    \"\"\"Collapse all whitespace runs into single spaces and trim the result.\n"
            "    \n"
            "    Example:\n"
            "    normalize_whitespace('  red\\n blue   green ') == 'red blue green'\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_count_vowels",
        (
            "def count_vowels(text: str) -> int:\n"
            "    \"\"\"Count vowels in text, ignoring case.\n"
            "    \n"
            "    Use a, e, i, o, and u as vowels.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_reverse_words",
        (
            "def reverse_words(text: str) -> str:\n"
            "    \"\"\"Return text with the word order reversed.\n"
            "    \n"
            "    Example:\n"
            "    reverse_words('red green blue') == 'blue green red'\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_is_palindrome",
        (
            "def is_palindrome(text: str) -> bool:\n"
            "    \"\"\"Return True if text is a palindrome after ignoring spaces and case.\n"
            "    \n"
            "    Example:\n"
            "    is_palindrome('Never odd or even') == True\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_title_case_words",
        (
            "def title_case_words(text: str) -> str:\n"
            "    \"\"\"Return text with the first letter of each word capitalized.\n"
            "    \n"
            "    Example:\n"
            "    title_case_words('small blue box') == 'Small Blue Box'\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_mask_email",
        (
            "def mask_email(email: str) -> str:\n"
            "    \"\"\"Hide the local part of an email except its first character.\n"
            "    \n"
            "    Example:\n"
            "    mask_email('alex@example.com') == 'a***@example.com'\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_extract_domain",
        (
            "def extract_domain(email: str) -> str:\n"
            "    \"\"\"Return the domain part of an email address.\n"
            "    \n"
            "    Example:\n"
            "    extract_domain('maya@school.edu') == 'school.edu'\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_slugify",
        (
            "def slugify(title: str) -> str:\n"
            "    \"\"\"Return a lowercase URL slug made from words in title.\n"
            "    \n"
            "    Replace whitespace with hyphens and drop non-alphanumeric characters.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_first_non_repeating_char",
        (
            "def first_non_repeating_char(text: str) -> str | None:\n"
            "    \"\"\"Return the first character that appears exactly once in text.\n"
            "    \n"
            "    Return None if every character repeats.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_longest_common_prefix",
        (
            "def longest_common_prefix(words: list[str]) -> str:\n"
            "    \"\"\"Return the longest string prefix shared by all words.\n"
            "    \n"
            "    Return an empty string when words is empty.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_dedupe_preserve_order",
        (
            "def dedupe_preserve_order(items: list[int]) -> list[int]:\n"
            "    \"\"\"Return items with duplicates removed while preserving first occurrence order.\n"
            "    \n"
            "    Example:\n"
            "    dedupe_preserve_order([3, 1, 3, 2, 1]) == [3, 1, 2]\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_flatten_once",
        (
            "def flatten_once(rows: list[list[int]]) -> list[int]:\n"
            "    \"\"\"Flatten one level of a list of integer lists.\n"
            "    \n"
            "    Example:\n"
            "    flatten_once([[1, 2], [], [3]]) == [1, 2, 3]\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_chunks",
        (
            "def chunks(items: list[int], size: int) -> list[list[int]]:\n"
            "    \"\"\"Split items into consecutive chunks of at most size elements.\n"
            "    \n"
            "    Example:\n"
            "    chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_running_total",
        (
            "def running_total(numbers: list[int]) -> list[int]:\n"
            "    \"\"\"Return the cumulative sum after each item in numbers.\n"
            "    \n"
            "    Example:\n"
            "    running_total([2, 5, -1]) == [2, 7, 6]\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_rotate_left",
        (
            "def rotate_left(items: list[int], steps: int) -> list[int]:\n"
            "    \"\"\"Return a new list with items rotated left by steps positions.\n"
            "    \n"
            "    The original list should not be modified.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_find_second_largest",
        (
            "def find_second_largest(numbers: list[int]) -> int | None:\n"
            "    \"\"\"Return the second largest distinct integer in numbers.\n"
            "    \n"
            "    Return None if fewer than two distinct values are present.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_binary_search",
        (
            "def binary_search(sorted_numbers: list[int], target: int) -> int:\n"
            "    \"\"\"Return the index of target in sorted_numbers, or -1 if absent.\n"
            "    \n"
            "    Assume sorted_numbers is sorted in ascending order.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_merge_sorted",
        (
            "def merge_sorted(left: list[int], right: list[int]) -> list[int]:\n"
            "    \"\"\"Merge two ascending sorted integer lists into one sorted list.\n"
            "    \n"
            "    Do not call sorted on the combined input.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_transpose_matrix",
        (
            "def transpose_matrix(matrix: list[list[int]]) -> list[list[int]]:\n"
            "    \"\"\"Return the transpose of a rectangular matrix.\n"
            "    \n"
            "    Example:\n"
            "    transpose_matrix([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_pairwise_sum",
        (
            "def pairwise_sum(a: list[int], b: list[int]) -> list[int]:\n"
            "    \"\"\"Return elementwise sums for two lists of equal length.\n"
            "    \n"
            "    Example:\n"
            "    pairwise_sum([1, 2, 3], [4, 5, 6]) == [5, 7, 9]\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_mode",
        (
            "def mode(numbers: list[int]) -> int | None:\n"
            "    \"\"\"Return the most frequent number in numbers.\n"
            "    \n"
            "    If there is a tie, return the smallest tied number. Return None for an empty list.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_remove_none",
        (
            "def remove_none(items: list[int | None]) -> list[int]:\n"
            "    \"\"\"Return a list containing only the non-None integers from items.\n"
            "    \n"
            "    Preserve the original order.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_partition_even_odd",
        (
            "def partition_even_odd(numbers: list[int]) -> tuple[list[int], list[int]]:\n"
            "    \"\"\"Return a tuple of two lists: even numbers first, odd numbers second.\n"
            "    \n"
            "    Preserve order inside each returned list.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_top_k",
        (
            "def top_k(numbers: list[int], k: int) -> list[int]:\n"
            "    \"\"\"Return the k largest numbers sorted from largest to smallest.\n"
            "    \n"
            "    If k is larger than the list length, return all numbers sorted descending.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_all_unique",
        (
            "def all_unique(items: list[str]) -> bool:\n"
            "    \"\"\"Return True if every string in items appears only once.\n"
            "    \n"
            "    Example:\n"
            "    all_unique(['a', 'b', 'a']) == False\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_invert_dict",
        (
            "def invert_dict(mapping: dict[str, str]) -> dict[str, str]:\n"
            "    \"\"\"Return a dictionary with keys and values swapped.\n"
            "    \n"
            "    Assume all values in mapping are unique.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_merge_counts",
        (
            "def merge_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:\n"
            "    \"\"\"Merge two dictionaries of integer counts by summing shared keys.\n"
            "    \n"
            "    The input dictionaries should not be modified.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_group_by_first_letter",
        (
            "def group_by_first_letter(words: list[str]) -> dict[str, list[str]]:\n"
            "    \"\"\"Group non-empty words by their first letter.\n"
            "    \n"
            "    Preserve the order of words inside each group.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_count_by_length",
        (
            "def count_by_length(words: list[str]) -> dict[int, int]:\n"
            "    \"\"\"Return a dictionary mapping word length to number of words with that length.\n"
            "    \n"
            "    Example:\n"
            "    count_by_length(['to', 'be', 'cat']) == {2: 2, 3: 1}\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_most_common_key",
        (
            "def most_common_key(counts: dict[str, int]) -> str | None:\n"
            "    \"\"\"Return the key with the largest count value.\n"
            "    \n"
            "    If there is a tie, return the alphabetically smallest key.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_filter_by_value",
        (
            "def filter_by_value(scores: dict[str, int], minimum: int) -> dict[str, int]:\n"
            "    \"\"\"Return entries whose values are greater than or equal to minimum.\n"
            "    \n"
            "    Preserve the original insertion order.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_dict_diff",
        (
            "def dict_diff(before: dict[str, int], after: dict[str, int]) -> dict[str, tuple[int | None, int | None]]:\n"
            "    \"\"\"Return changed, added, and removed keys between before and after.\n"
            "    \n"
            "    Each result value should be a tuple of (old_value, new_value).\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_nested_get",
        (
            "def nested_get(data: dict[str, dict[str, int]], outer: str, inner: str, default: int) -> int:\n"
            "    \"\"\"Return data[outer][inner] when both keys exist, otherwise default.\n"
            "    \n"
            "    Do not mutate data.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_total_by_category",
        (
            "def total_by_category(rows: list[dict[str, int | str]]) -> dict[str, int]:\n"
            "    \"\"\"Sum row['amount'] values grouped by row['category'].\n"
            "    \n"
            "    Assume each row has a string category and integer amount.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_index_by_id",
        (
            "def index_by_id(rows: list[dict[str, int | str]]) -> dict[int, dict[str, int | str]]:\n"
            "    \"\"\"Return a dictionary mapping each row's integer id to the row itself.\n"
            "    \n"
            "    Assume every row contains an id key.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_parse_csv_line",
        (
            "def parse_csv_line(line: str) -> list[str]:\n"
            "    \"\"\"Parse a simple comma-separated line without quoted fields.\n"
            "    \n"
            "    Strip surrounding whitespace from each field.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_format_table_row",
        (
            "def format_table_row(values: list[str], width: int) -> str:\n"
            "    \"\"\"Return values padded on the right to width and joined by ' | '.\n"
            "    \n"
            "    Example:\n"
            "    format_table_row(['A', 'cat'], 4) == 'A    | cat '\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_parse_query_string",
        (
            "def parse_query_string(query: str) -> dict[str, str]:\n"
            "    \"\"\"Parse a URL query string like 'a=1&b=two' into a dictionary.\n"
            "    \n"
            "    Do not perform percent-decoding.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_parse_int_list",
        (
            "def parse_int_list(text: str) -> list[int]:\n"
            "    \"\"\"Parse comma-separated integers from text.\n"
            "    \n"
            "    Ignore empty fields after stripping whitespace.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_hex_to_rgb",
        (
            "def hex_to_rgb(color: str) -> tuple[int, int, int]:\n"
            "    \"\"\"Convert a hex color like '#1a2b3c' or '1a2b3c' to an RGB tuple.\n"
            "    \n"
            "    Assume the input contains exactly six hexadecimal digits after optional '#'.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_format_phone_number",
        (
            "def format_phone_number(digits: str) -> str:\n"
            "    \"\"\"Format a 10-digit string as '(XXX) XXX-XXXX'.\n"
            "    \n"
            "    Example:\n"
            "    format_phone_number('4155550123') == '(415) 555-0123'\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_parse_log_level",
        (
            "def parse_log_level(line: str) -> str | None:\n"
            "    \"\"\"Return the log level from a line starting with '[LEVEL]'.\n"
            "    \n"
            "    Return None if the line does not start with a bracketed level.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_key_value_lines_to_dict",
        (
            "def key_value_lines_to_dict(text: str) -> dict[str, str]:\n"
            "    \"\"\"Parse lines of the form 'key=value' into a dictionary.\n"
            "    \n"
            "    Skip blank lines and strip whitespace around keys and values.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_render_markdown_link",
        (
            "def render_markdown_link(label: str, url: str) -> str:\n"
            "    \"\"\"Return a Markdown link for label and url.\n"
            "    \n"
            "    Example:\n"
            "    render_markdown_link('Docs', 'https://example.com') == '[Docs](https://example.com)'\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_snake_to_camel",
        (
            "def snake_to_camel(name: str) -> str:\n"
            "    \"\"\"Convert a snake_case name to lower camelCase.\n"
            "    \n"
            "    Example:\n"
            "    snake_to_camel('user_profile_id') == 'userProfileId'\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_is_leap_year",
        (
            "def is_leap_year(year: int) -> bool:\n"
            "    \"\"\"Return True if year is a leap year in the Gregorian calendar.\n"
            "    \n"
            "    Years divisible by 100 are leap years only when also divisible by 400.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_days_in_month",
        (
            "def days_in_month(year: int, month: int) -> int:\n"
            "    \"\"\"Return the number of days in the given month of the given year.\n"
            "    \n"
            "    Use leap-year rules for February.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_add_minutes_to_time",
        (
            "def add_minutes_to_time(time_text: str, minutes: int) -> str:\n"
            "    \"\"\"Add minutes to a 24-hour HH:MM time and wrap around midnight.\n"
            "    \n"
            "    Return the result as HH:MM.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_minutes_between_times",
        (
            "def minutes_between_times(start: str, end: str) -> int:\n"
            "    \"\"\"Return minutes from start HH:MM to end HH:MM on the same day.\n"
            "    \n"
            "    Assume end is not earlier than start.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_next_weekday",
        (
            "def next_weekday(current: str) -> str:\n"
            "    \"\"\"Return the weekday name after current.\n"
            "    \n"
            "    Use Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_file_extension",
        (
            "def file_extension(path: str) -> str:\n"
            "    \"\"\"Return the file extension after the final dot, without the dot.\n"
            "    \n"
            "    Return an empty string if there is no extension.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_basename_without_extension",
        (
            "def basename_without_extension(path: str) -> str:\n"
            "    \"\"\"Return the final path component without its final extension.\n"
            "    \n"
            "    Both '/' and '\\\\' may appear as separators.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_join_url_parts",
        (
            "def join_url_parts(base: str, path: str) -> str:\n"
            "    \"\"\"Join a base URL and path with exactly one slash between them.\n"
            "    \n"
            "    Preserve any trailing slash already present in path.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_normalize_path_parts",
        (
            "def normalize_path_parts(parts: list[str]) -> str:\n"
            "    \"\"\"Join path parts with single '/' separators.\n"
            "    \n"
            "    Ignore empty parts and strip leading or trailing slashes from each part.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_is_hidden_filename",
        (
            "def is_hidden_filename(filename: str) -> bool:\n"
            "    \"\"\"Return True if filename represents a hidden Unix-style file.\n"
            "    \n"
            "    A hidden filename starts with '.' but is not '.' or '..'.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_balanced_parentheses",
        (
            "def balanced_parentheses(text: str) -> bool:\n"
            "    \"\"\"Return True if parentheses in text are balanced.\n"
            "    \n"
            "    Ignore all characters except '(' and ')'.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_evaluate_rpn",
        (
            "def evaluate_rpn(tokens: list[str]) -> int:\n"
            "    \"\"\"Evaluate a reverse Polish notation expression with integer operands.\n"
            "    \n"
            "    Support '+', '-', '*', and '/' with truncation toward zero.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_adjacent_differences",
        (
            "def adjacent_differences(numbers: list[int]) -> list[int]:\n"
            "    \"\"\"Return differences numbers[i + 1] - numbers[i] for adjacent pairs.\n"
            "    \n"
            "    Return an empty list when there are fewer than two numbers.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_cumulative_max",
        (
            "def cumulative_max(numbers: list[int]) -> list[int]:\n"
            "    \"\"\"Return the maximum value seen so far at each position.\n"
            "    \n"
            "    Example:\n"
            "    cumulative_max([3, 1, 5, 2]) == [3, 3, 5, 5]\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_sliding_window_max",
        (
            "def sliding_window_max(numbers: list[int], size: int) -> list[int]:\n"
            "    \"\"\"Return the maximum for each consecutive window of length size.\n"
            "    \n"
            "    Return an empty list if size is larger than the input length.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_longest_increasing_run",
        (
            "def longest_increasing_run(numbers: list[int]) -> int:\n"
            "    \"\"\"Return the length of the longest strictly increasing contiguous run.\n"
            "    \n"
            "    Return 0 for an empty list.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_find_anagrams",
        (
            "def find_anagrams(words: list[str], target: str) -> list[str]:\n"
            "    \"\"\"Return words that are anagrams of target.\n"
            "    \n"
            "    Do not include target itself when it appears with the same spelling.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_two_sum_indices",
        (
            "def two_sum_indices(numbers: list[int], target: int) -> tuple[int, int] | None:\n"
            "    \"\"\"Return indices of two distinct numbers that add up to target.\n"
            "    \n"
            "    Return None if no pair exists.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_compress_runs",
        (
            "def compress_runs(text: str) -> list[tuple[str, int]]:\n"
            "    \"\"\"Return run-length encoding pairs for consecutive equal characters.\n"
            "    \n"
            "    Example:\n"
            "    compress_runs('aaabbc') == [('a', 3), ('b', 2), ('c', 1)]\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_expand_runs",
        (
            "def expand_runs(runs: list[tuple[str, int]]) -> str:\n"
            "    \"\"\"Expand run-length encoding pairs back into a string.\n"
            "    \n"
            "    Example:\n"
            "    expand_runs([('a', 3), ('b', 2)]) == 'aaabb'\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_interval_overlap",
        (
            "def interval_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:\n"
            "    \"\"\"Return True if two closed integer intervals overlap.\n"
            "    \n"
            "    Each interval is represented as (start, end).\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_merge_intervals",
        (
            "def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
            "    \"\"\"Merge overlapping closed intervals and return them sorted by start.\n"
            "    \n"
            "    Example:\n"
            "    merge_intervals([(1, 3), (2, 5), (8, 9)]) == [(1, 5), (8, 9)]\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_shortest_path_length",
        (
            "def shortest_path_length(graph: dict[str, list[str]], start: str, goal: str) -> int | None:\n"
            "    \"\"\"Return the fewest number of edges from start to goal in an unweighted graph.\n"
            "    \n"
            "    Return None when goal is unreachable.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_adjacency_from_edges",
        (
            "def adjacency_from_edges(edges: list[tuple[str, str]]) -> dict[str, list[str]]:\n"
            "    \"\"\"Build an undirected adjacency list from edge pairs.\n"
            "    \n"
            "    Preserve neighbor insertion order for each node.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_has_cycle_undirected",
        (
            "def has_cycle_undirected(graph: dict[str, list[str]]) -> bool:\n"
            "    \"\"\"Return True if an undirected graph contains a cycle.\n"
            "    \n"
            "    The graph is represented as an adjacency list.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_validate_password",
        (
            "def validate_password(password: str) -> bool:\n"
            "    \"\"\"Return True if password has at least 8 characters, a digit, and a letter.\n"
            "    \n"
            "    Whitespace is allowed but still counts as a character.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_is_valid_ipv4",
        (
            "def is_valid_ipv4(address: str) -> bool:\n"
            "    \"\"\"Return True if address is a valid dotted IPv4 address.\n"
            "    \n"
            "    Each part must be an integer from 0 through 255 without empty parts.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_is_valid_email_simple",
        (
            "def is_valid_email_simple(email: str) -> bool:\n"
            "    \"\"\"Return True for a simple email containing one '@' and a dotted domain.\n"
            "    \n"
            "    The local part and each domain part must be non-empty.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_parse_bool",
        (
            "def parse_bool(text: str) -> bool | None:\n"
            "    \"\"\"Parse common boolean strings into True or False.\n"
            "    \n"
            "    Accept true, yes, 1, false, no, and 0 ignoring case. Return None otherwise.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_safe_divide",
        (
            "def safe_divide(a: float, b: float) -> float | None:\n"
            "    \"\"\"Return a / b, or None when b is zero.\n"
            "    \n"
            "    Do not raise ZeroDivisionError.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_median",
        (
            "def median(numbers: list[float]) -> float | None:\n"
            "    \"\"\"Return the median of numbers.\n"
            "    \n"
            "    Return None for an empty list. Do not mutate the input list.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_matrix_diagonal_sum",
        (
            "def matrix_diagonal_sum(matrix: list[list[int]]) -> int:\n"
            "    \"\"\"Return the sum of the main diagonal of a square matrix.\n"
            "    \n"
            "    Example:\n"
            "    matrix_diagonal_sum([[1, 2], [3, 4]]) == 5\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_spiral_order",
        (
            "def spiral_order(matrix: list[list[int]]) -> list[int]:\n"
            "    \"\"\"Return the elements of a rectangular matrix in clockwise spiral order.\n"
            "    \n"
            "    Return an empty list for an empty matrix.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_tic_tac_toe_winner",
        (
            "def tic_tac_toe_winner(board: list[list[str]]) -> str | None:\n"
            "    \"\"\"Return 'X' or 'O' if that player has won a 3x3 tic-tac-toe board.\n"
            "    \n"
            "    Return None when there is no winner.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_is_valid_sudoku_row",
        (
            "def is_valid_sudoku_row(row: list[int]) -> bool:\n"
            "    \"\"\"Return True if row contains no repeated nonzero values from 1 to 9.\n"
            "    \n"
            "    Zeros represent empty cells.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_add_polynomials",
        (
            "def add_polynomials(a: list[int], b: list[int]) -> list[int]:\n"
            "    \"\"\"Add two polynomials represented by coefficient lists.\n"
            "    \n"
            "    Coefficient index is the power of x. Trim trailing zeros from the result.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_dot_product",
        (
            "def dot_product(a: list[float], b: list[float]) -> float:\n"
            "    \"\"\"Return the dot product of two equal-length numeric lists.\n"
            "    \n"
            "    Example:\n"
            "    dot_product([1, 2, 3], [4, 5, 6]) == 32\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_scale_values",
        (
            "def scale_values(values: list[float], factor: float) -> list[float]:\n"
            "    \"\"\"Return a new list with each value multiplied by factor.\n"
            "    \n"
            "    Do not mutate the input list.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_normalize_scores",
        (
            "def normalize_scores(scores: list[float]) -> list[float]:\n"
            "    \"\"\"Normalize scores to the range 0.0 through 1.0.\n"
            "    \n"
            "    If all scores are equal, return 0.0 for every score.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_weighted_average",
        (
            "def weighted_average(values: list[float], weights: list[float]) -> float | None:\n"
            "    \"\"\"Return the weighted average of values.\n"
            "    \n"
            "    Return None when the inputs are empty or the sum of weights is zero.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_cartesian_product",
        (
            "def cartesian_product(left: list[str], right: list[str]) -> list[tuple[str, str]]:\n"
            "    \"\"\"Return all ordered pairs from left and right.\n"
            "    \n"
            "    Preserve input order with left as the outer loop.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_powerset",
        (
            "def powerset(items: list[str]) -> list[list[str]]:\n"
            "    \"\"\"Return all subsets of items as lists.\n"
            "    \n"
            "    Preserve item order inside each subset.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_find_peaks",
        (
            "def find_peaks(numbers: list[int]) -> list[int]:\n"
            "    \"\"\"Return indices whose values are greater than both immediate neighbors.\n"
            "    \n"
            "    The first and last elements are not peaks.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_drop_every_nth",
        (
            "def drop_every_nth(items: list[str], n: int) -> list[str]:\n"
            "    \"\"\"Return items with every nth element removed using one-based counting.\n"
            "    \n"
            "    Example:\n"
            "    drop_every_nth(['a', 'b', 'c', 'd'], 2) == ['a', 'c']\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_interleave_lists",
        (
            "def interleave_lists(a: list[int], b: list[int]) -> list[int]:\n"
            "    \"\"\"Return a list alternating items from a and b.\n"
            "    \n"
            "    Append any leftover items from the longer list at the end.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_count_substring",
        (
            "def count_substring(text: str, needle: str) -> int:\n"
            "    \"\"\"Count non-overlapping occurrences of needle in text.\n"
            "    \n"
            "    Return 0 when needle is empty.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_replace_at_indices",
        (
            "def replace_at_indices(text: str, indices: set[int], replacement: str) -> str:\n"
            "    \"\"\"Return text with characters at the given indices replaced.\n"
            "    \n"
            "    Ignore indices outside the string.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_line_lengths",
        (
            "def line_lengths(text: str) -> list[int]:\n"
            "    \"\"\"Return the length of each line in text.\n"
            "    \n"
            "    Use str.splitlines() behavior for handling line breaks.\n"
            "    \"\"\"\n"
        ),
    ),
    (
        "python_complete_strip_comments",
        (
            "def strip_comments(lines: list[str]) -> list[str]:\n"
            "    \"\"\"Return lines with everything after a '#' character removed.\n"
            "    \n"
            "    Strip trailing whitespace from each returned line.\n"
            "    \"\"\"\n"
        ),
    ),
]

if not 95 <= len(INFERENCE_TEST_PROMPTS_PYTHON_COMPLETION) <= 105:
    raise ValueError("Expected around 100 Python completion prompts.")


INFERENCE_TEST_PROMPTS_PYTHON_DIVERSE = [
    (
        "python_diverse_lru_cache",
        (
            "# Implement a tiny LRU cache\n"
            "# cache maps keys to (value, last_used_tick).\n"
            "# Return a new cache after inserting or updating key.\n"
            "# When capacity is exceeded, evict the least recently used key.\n"
            "# Break ties by evicting the alphabetically smallest key.\n"
            "def lru_cache_put(cache: dict[str, tuple[int, int]], capacity: int, key: str, value: int, tick: int) -> dict[str, tuple[int, int]]:\n"
        ),
    ),
    (
        "python_diverse_topological_layers",
        (
            "# Group a dependency graph into topological layers\n"
            "# deps maps each task to the set of tasks it depends on.\n"
            "# Each returned layer contains tasks whose dependencies are already done.\n"
            "# Sort task names inside each layer.\n"
            "# Raise ValueError if the graph contains a cycle.\n"
            "def topological_layers(deps: dict[str, set[str]]) -> list[list[str]]:\n"
        ),
    ),
    (
        "python_diverse_min_window_substring",
        (
            "# Find a minimum covering substring\n"
            "# Return the shortest substring of text containing all characters in required.\n"
            "# Character multiplicity matters.\n"
            "# Return an empty string when no window exists.\n"
            "# Prefer the earliest window when multiple windows have the same length.\n"
            "def min_window_substring(text: str, required: str) -> str:\n"
        ),
    ),
    (
        "python_diverse_edit_distance",
        (
            "# Compute edit distance with dynamic programming\n"
            "# Return the Levenshtein distance between a and b.\n"
            "# Allowed operations are insert, delete, and replace.\n"
            "# All operations have cost 1.\n"
            "def edit_distance(a: str, b: str) -> int:\n"
        ),
    ),
    (
        "python_diverse_knapsack",
        (
            "# Solve a small 0/1 knapsack problem\n"
            "# Each item is (weight, value).\n"
            "# Return the maximum total value without exceeding capacity.\n"
            "# Each item may be used at most once.\n"
            "def knapsack_max_value(items: list[tuple[int, int]], capacity: int) -> int:\n"
        ),
    ),
    (
        "python_diverse_expression_tokens",
        (
            "# Tokenize a tiny arithmetic language\n"
            "# Return integer, identifier, operator, and parenthesis tokens.\n"
            "# Operators are +, -, *, /, and =.\n"
            "# Ignore whitespace.\n"
            "# Raise ValueError on any unexpected character.\n"
            "def tokenize_expression(expr: str) -> list[str]:\n"
        ),
    ),
    (
        "python_diverse_match_brackets",
        (
            "# Validate mixed bracket pairs\n"
            "# Support (), [], and {}.\n"
            "# Ignore all other characters.\n"
            "# Return True only when every opener is closed in the correct order.\n"
            "def validate_brackets(text: str) -> bool:\n"
        ),
    ),
    (
        "python_diverse_rate_limit",
        (
            "# Evaluate a sliding-window rate limit\n"
            "# timestamps are nondecreasing seconds.\n"
            "# For each timestamp, return whether the event is allowed.\n"
            "# An event is allowed if fewer than limit allowed events occurred in the previous window seconds.\n"
            "# Allowed events count toward future limits.\n"
            "def allowed_events(timestamps: list[int], limit: int, window: int) -> list[bool]:\n"
        ),
    ),
    (
        "python_diverse_diff_lines",
        (
            "# Produce a compact line diff\n"
            "# Return lines prefixed with '  ' for unchanged, '- ' for removed, and '+ ' for added.\n"
            "# Use a longest-common-subsequence strategy.\n"
            "# Keep the output deterministic when several diffs are possible.\n"
            "def simple_line_diff(before: list[str], after: list[str]) -> list[str]:\n"
        ),
    ),
    (
        "python_diverse_json_pointer",
        (
            "# Resolve a JSON Pointer path\n"
            "# Implement RFC-style slash-separated JSON pointer traversal.\n"
            "# Support dict keys and list indices.\n"
            "# Decode ~1 as / and ~0 as ~.\n"
            "# Raise KeyError or IndexError for missing targets.\n"
            "def resolve_json_pointer(document: object, pointer: str) -> object:\n"
        ),
    ),
    (
        "python_diverse_invoice_tax",
        (
            "# Summarize invoice rows with discounts and tax\n"
            "# Each row has quantity, unit_price, and discount_percent.\n"
            "# Apply discount per row before tax.\n"
            "# Return the final total rounded to 2 decimals.\n"
            "def invoice_total(rows: list[dict[str, float]], tax_rate: float) -> float:\n"
        ),
    ),
    (
        "python_diverse_roman_numerals",
        (
            "# Convert integers to Roman numerals\n"
            "# Support numbers from 1 through 3999.\n"
            "# Use subtractive notation such as IV, IX, XL, and CM.\n"
            "# Raise ValueError outside the supported range.\n"
            "def to_roman(number: int) -> str:\n"
        ),
    ),
    (
        "python_diverse_parse_semver",
        (
            "# Parse and compare semantic versions\n"
            "# Return -1, 0, or 1 depending on version ordering.\n"
            "# Compare major, minor, and patch numerically.\n"
            "# Ignore build metadata after '+'.\n"
            "# Pre-release versions sort before normal versions.\n"
            "def compare_semver(left: str, right: str) -> int:\n"
        ),
    ),
    (
        "python_diverse_calendar_grid",
        (
            "# Create a month calendar grid\n"
            "# first_weekday uses Monday=0 through Sunday=6.\n"
            "# Return weeks of length 7.\n"
            "# Use None for cells outside the month.\n"
            "def month_grid(first_weekday: int, days_in_month: int) -> list[list[int | None]]:\n"
        ),
    ),
    (
        "python_diverse_sessionize",
        (
            "# Group events into activity sessions\n"
            "# Each event is (user_id, timestamp).\n"
            "# For each user, start a new session after a gap greater than timeout.\n"
            "# Return timestamp sessions in chronological order per user.\n"
            "def sessionize(events: list[tuple[str, int]], timeout: int) -> dict[str, list[list[int]]]:\n"
        ),
    ),
    (
        "python_diverse_markdown_toc",
        (
            "# Build a Markdown table of contents\n"
            "# Find ATX headings beginning with one to six # characters.\n"
            "# Return (level, title, slug) tuples.\n"
            "# Slugify by lowercasing, keeping alphanumerics and spaces, and replacing spaces with hyphens.\n"
            "def markdown_toc(markdown: str) -> list[tuple[int, str, str]]:\n"
        ),
    ),
    (
        "python_diverse_csv_records",
        (
            "# Parse simple CSV records with quotes\n"
            "# Support comma-separated fields and double-quoted fields.\n"
            "# Inside quoted fields, two double quotes represent one literal quote.\n"
            "# Rows are separated by newlines.\n"
            "def parse_csv_records(text: str) -> list[list[str]]:\n"
        ),
    ),
    (
        "python_diverse_acl_decision",
        (
            "# Evaluate allow/deny access rules\n"
            "# Rules contain effect, user, and resource.\n"
            "# A rule field value of '*' matches anything.\n"
            "# Later matching rules override earlier matching rules.\n"
            "# Default is deny.\n"
            "def access_allowed(rules: list[dict[str, str]], user: str, resource: str) -> bool:\n"
        ),
    ),
    (
        "python_diverse_binary_tree_width",
        (
            "# Compute the maximum width of a sparse binary tree\n"
            "# level_order is a heap-style list where None means a missing node.\n"
            "# Width is measured between the leftmost and rightmost non-missing nodes at a level.\n"
            "# Return 0 for an empty tree.\n"
            "def max_tree_width(level_order: list[int | None]) -> int:\n"
        ),
    ),
    (
        "python_diverse_word_ladder",
        (
            "# Find the length of a word ladder\n"
            "# Each step changes exactly one character.\n"
            "# Intermediate words must be in words.\n"
            "# Return the number of words in the shortest ladder including start and goal.\n"
            "# Return None if no ladder exists.\n"
            "def word_ladder_length(start: str, goal: str, words: set[str]) -> int | None:\n"
        ),
    ),
    (
        "python_diverse_class_priority_queue",
        (
            "# Complete a stable priority queue\n"
            "class StablePriorityQueue:\n"
            "    # Implement push(item: str, priority: int) -> None.\n"
            "    # Implement pop() -> str.\n"
            "    # Lower priority numbers are popped first.\n"
            "    # Items with equal priority are popped in insertion order.\n"
            "    # Raise IndexError when popping from an empty queue.\n"
        ),
    ),
    (
        "python_diverse_class_rolling_stats",
        (
            "# Complete rolling statistics over a fixed-size window\n"
            "class RollingStats:\n"
            "    # The constructor takes window_size: int.\n"
            "    # add(value: float) records a value and drops the oldest value if needed.\n"
            "    # mean() returns the current mean or None when empty.\n"
            "    # variance() returns population variance or None when empty.\n"
        ),
    ),
    (
        "python_diverse_class_token_bucket",
        (
            "# Complete a token bucket rate limiter\n"
            "class TokenBucket:\n"
            "    # The constructor takes capacity: int and refill_per_second: float.\n"
            "    # allow(now: float, cost: float = 1.0) returns True if enough tokens exist.\n"
            "    # Refill lazily based on elapsed time.\n"
            "    # Tokens may never exceed capacity.\n"
        ),
    ),
    (
        "python_diverse_class_inventory",
        (
            "# Complete inventory reservation logic\n"
            "class Inventory:\n"
            "    # Track stock and reserved counts per sku.\n"
            "    # add_stock(sku: str, quantity: int) increases stock.\n"
            "    # reserve(sku: str, quantity: int) returns True only if available stock is enough.\n"
            "    # commit(sku: str, quantity: int) removes reserved items.\n"
            "    # release(sku: str, quantity: int) unreserves items.\n"
        ),
    ),
    (
        "python_diverse_class_undo_stack",
        (
            "# Complete a bounded undo stack\n"
            "class UndoStack:\n"
            "    # The constructor takes max_size: int.\n"
            "    # push(state: str) records a state and clears redo history.\n"
            "    # undo() moves back one state and returns it.\n"
            "    # redo() moves forward one state and returns it.\n"
            "    # Return None when undo or redo is not possible.\n"
        ),
    ),
    (
        "python_diverse_class_graph",
        (
            "# Complete a small directed graph helper\n"
            "class DirectedGraph:\n"
            "    # add_edge(source: str, target: str) records an edge.\n"
            "    # neighbors(node: str) returns sorted outgoing neighbors.\n"
            "    # reachable(start: str) returns all reachable nodes excluding start, sorted alphabetically.\n"
        ),
    ),
    (
        "python_diverse_class_ttl_cache",
        (
            "# Complete a TTL cache\n"
            "class TTLCache:\n"
            "    # The constructor takes ttl_seconds: int.\n"
            "    # set(key: str, value: str, now: int) stores a value with expiry.\n"
            "    # get(key: str, now: int) returns the value or None when missing or expired.\n"
            "    # purge(now: int) removes expired entries.\n"
        ),
    ),
    (
        "python_diverse_class_bank_account",
        (
            "# Complete a bank account with ledger entries\n"
            "class BankAccount:\n"
            "    # deposit(amount: int, memo: str) and withdraw(amount: int, memo: str) update balance.\n"
            "    # Reject negative amounts.\n"
            "    # withdraw returns False without changing balance when funds are insufficient.\n"
            "    # statement() returns ledger rows in insertion order.\n"
        ),
    ),
    (
        "python_diverse_class_markdown_builder",
        (
            "# Complete a tiny Markdown builder\n"
            "class MarkdownBuilder:\n"
            "    # heading(text: str, level: int = 1) appends a heading.\n"
            "    # paragraph(text: str) appends a paragraph.\n"
            "    # bullet(items: list[str]) appends a bullet list.\n"
            "    # render() returns the document with blank lines between blocks.\n"
        ),
    ),
    (
        "python_diverse_class_range_set",
        (
            "# Complete an integer range set\n"
            "class RangeSet:\n"
            "    # add(start: int, end: int) adds an inclusive range.\n"
            "    # Overlapping or adjacent ranges should be merged.\n"
            "    # contains(value: int) returns whether value is covered.\n"
            "    # ranges() returns merged ranges sorted by start.\n"
        ),
    ),
    (
        "python_diverse_bugfix_group_runs",
        (
            "# Fix the implementation below: group consecutive equal values\n"
            "# The current version emits an invalid first run and misses the final run.\n"
            "# Return [] for an empty input.\n"
            "\n"
            "def group_runs(values: list[str]) -> list[tuple[str, int]]:\n"
            "    runs = []\n"
            "    current = None\n"
            "    count = 0\n"
            "    for value in values:\n"
            "        if value == current:\n"
            "            count += 1\n"
            "        else:\n"
            "            runs.append((current, count))\n"
            "            current = value\n"
            "            count = 1\n"
            "    return runs\n"
        ),
    ),
    (
        "python_diverse_bugfix_merge_defaults",
        (
            "# Fix the implementation below: merge nested configuration dictionaries\n"
            "# The current version mutates defaults.\n"
            "# Nested dictionaries should be merged recursively.\n"
            "# Return a new dictionary.\n"
            "\n"
            "def merge_config(defaults: dict, overrides: dict) -> dict:\n"
            "    result = defaults\n"
            "    for key, value in overrides.items():\n"
            "        if isinstance(value, dict):\n"
            "            result[key] = merge_config(result.get(key, {}), value)\n"
            "        else:\n"
            "            result[key] = value\n"
            "    return result\n"
        ),
    ),
    (
        "python_diverse_bugfix_chunk_text",
        (
            "# Fix the implementation below: split text into bounded chunks without losing words\n"
            "# Avoid leading spaces in chunks.\n"
            "# A single long word may exceed max_chars by itself.\n"
            "# Return [] when words is empty.\n"
            "\n"
            "def chunk_text(words: list[str], max_chars: int) -> list[str]:\n"
            "    chunks = []\n"
            "    current = ''\n"
            "    for word in words:\n"
            "        if len(current) + len(word) > max_chars:\n"
            "            chunks.append(current)\n"
            "            current = word\n"
            "        else:\n"
            "            current += ' ' + word\n"
            "    chunks.append(current)\n"
            "    return chunks\n"
        ),
    ),
    (
        "python_diverse_bugfix_percentiles",
        (
            "# Fix the implementation below: compute nearest-rank percentiles\n"
            "# Do not mutate values.\n"
            "# p is between 0 and 100 inclusive.\n"
            "# Use nearest-rank indexing with p=100 returning the largest value.\n"
            "# Raise ValueError for an empty list.\n"
            "\n"
            "def percentile(values: list[float], p: float) -> float:\n"
            "    values.sort()\n"
            "    index = int(len(values) * p / 100)\n"
            "    return values[index]\n"
        ),
    ),
    (
        "python_diverse_bugfix_parse_ranges",
        (
            "# Fix the implementation below: parse comma-separated integer ranges\n"
            "# Ranges are inclusive.\n"
            "# Whitespace around parts is allowed.\n"
            "# Empty parts should be ignored.\n"
            "\n"
            "def parse_ranges(spec: str) -> set[int]:\n"
            "    result = set()\n"
            "    for part in spec.split(','):\n"
            "        if '-' in part:\n"
            "            start, end = part.split('-')\n"
            "            for value in range(int(start), int(end)):\n"
            "                result.add(value)\n"
            "        else:\n"
            "            result.add(int(part))\n"
            "    return result\n"
        ),
    ),
    (
        "python_diverse_bugfix_word_counts",
        (
            "# Fix the implementation below: count normalized words\n"
            "# The current version raises KeyError.\n"
            "# Skip empty tokens.\n"
            "# Treat runs of whitespace as separators.\n"
            "\n"
            "def word_counts(text: str) -> dict[str, int]:\n"
            "    counts = {}\n"
            "    for word in text.split(' '):\n"
            "        word = word.lower().strip('.,!?')\n"
            "        counts[word] += 1\n"
            "    return counts\n"
        ),
    ),
    (
        "python_diverse_bugfix_unique_paths",
        (
            "# Fix the implementation below: count grid paths with blocked cells\n"
            "# Cells with 1 are blocked and cells with 0 are open.\n"
            "# Moves may only go right or down.\n"
            "# Handle first row and first column correctly.\n"
            "\n"
            "def count_paths(grid: list[list[int]]) -> int:\n"
            "    rows = len(grid)\n"
            "    cols = len(grid[0])\n"
            "    dp = [[0] * cols] * rows\n"
            "    dp[0][0] = 1\n"
            "    for r in range(rows):\n"
            "        for c in range(cols):\n"
            "            if grid[r][c] == 1:\n"
            "                dp[r][c] = 0\n"
            "            else:\n"
            "                dp[r][c] += dp[r - 1][c] + dp[r][c - 1]\n"
            "    return dp[-1][-1]\n"
        ),
    ),
    (
        "python_diverse_bugfix_flatten_dict",
        (
            "# Fix the implementation below: flatten nested dictionaries with dotted keys\n"
            "# Top-level keys should not start with a dot.\n"
            "# Empty nested dictionaries should be preserved as values.\n"
            "\n"
            "def flatten_dict(data: dict, prefix: str = '') -> dict[str, object]:\n"
            "    result = {}\n"
            "    for key, value in data.items():\n"
            "        new_key = prefix + '.' + key\n"
            "        if isinstance(value, dict):\n"
            "            result.update(flatten_dict(value, new_key))\n"
            "        else:\n"
            "            result[new_key] = value\n"
            "    return result\n"
        ),
    ),
    (
        "python_diverse_bugfix_dedupe_records",
        (
            "# Fix the implementation below: deduplicate records by id keeping newest timestamp\n"
            "# Keep the record with the largest timestamp for each id.\n"
            "# Return records sorted by id.\n"
            "# Do not mutate the input records.\n"
            "\n"
            "def dedupe_records(records: list[dict]) -> list[dict]:\n"
            "    seen = {}\n"
            "    for record in records:\n"
            "        if record['id'] not in seen:\n"
            "            seen[record['id']] = record\n"
            "    return list(seen.values())\n"
        ),
    ),
    (
        "python_diverse_bugfix_safe_eval",
        (
            "# Fix the implementation below: evaluate a restricted arithmetic AST\n"
            "# Support number nodes and binary nodes.\n"
            "# Allowed operators are +, -, *, and /.\n"
            "# Raise ValueError for unknown node types or operators.\n"
            "# Use normal Python division for /.\n"
            "\n"
            "def eval_node(node):\n"
            "    if node['type'] == 'number':\n"
            "        return node['value']\n"
            "    left = eval_node(node['left'])\n"
            "    right = eval_node(node['right'])\n"
            "    if node['op'] == '+':\n"
            "        return left + right\n"
            "    if node['op'] == '-':\n"
            "        return left - right\n"
            "    return left * right\n"
        ),
    ),
    (
        "python_diverse_tests_reconcile",
        (
            "# Implement the function so all assertions pass: reconcile inventory transactions\n"
            "assert reconcile([('add', 'A', 5), ('remove', 'A', 2)]) == {'A': 3}\n"
            "assert reconcile([('remove', 'B', 1)]) == {'B': -1}\n"
            "\n"
            "def reconcile(transactions: list[tuple[str, str, int]]) -> dict[str, int]:\n"
        ),
    ),
    (
        "python_diverse_tests_windowed_average",
        (
            "# Implement the function so all assertions pass: compute trailing window averages\n"
            "assert windowed_average([2, 4, 6, 8], 2) == [2.0, 3.0, 5.0, 7.0]\n"
            "assert windowed_average([], 3) == []\n"
            "\n"
            "def windowed_average(values: list[float], size: int) -> list[float]:\n"
        ),
    ),
    (
        "python_diverse_tests_apply_patch_ops",
        (
            "# Implement the function so all assertions pass: apply simple patch operations to a list\n"
            "assert apply_ops(['a', 'b'], [('insert', 1, 'x'), ('delete', 0, '')]) == ['x', 'b']\n"
            "assert apply_ops([], [('insert', 0, 'z')]) == ['z']\n"
            "\n"
            "def apply_ops(items: list[str], ops: list[tuple[str, int, str]]) -> list[str]:\n"
        ),
    ),
    (
        "python_diverse_tests_summarize_grades",
        (
            "# Implement the function so all assertions pass: summarize grades by student\n"
            "rows = [{'name': 'Ada', 'score': 10}, {'name': 'Ada', 'score': 20}, {'name': 'Ben', 'score': 7}]\n"
            "assert summarize_grades(rows) == {'Ada': {'count': 2, 'average': 15.0}, 'Ben': {'count': 1, 'average': 7.0}}\n"
            "\n"
            "def summarize_grades(rows: list[dict[str, int | str]]) -> dict[str, dict[str, float | int]]:\n"
        ),
    ),
    (
        "python_diverse_tests_compact_ranges",
        (
            "# Implement the function so all assertions pass: compact sorted integers into ranges\n"
            "assert compact_ranges([1, 2, 3, 7, 9, 10]) == ['1-3', '7', '9-10']\n"
            "assert compact_ranges([]) == []\n"
            "\n"
            "def compact_ranges(numbers: list[int]) -> list[str]:\n"
        ),
    ),
    (
        "python_diverse_tests_resolve_aliases",
        (
            "# Implement the function so all assertions pass: resolve chained aliases\n"
            "assert resolve_aliases({'a': 'b', 'b': 'c', 'c': 'c'}) == {'a': 'c', 'b': 'c', 'c': 'c'}\n"
            "assert resolve_aliases({'x': 'y', 'y': 'x'}) == {'x': None, 'y': None}\n"
            "\n"
            "def resolve_aliases(aliases: dict[str, str]) -> dict[str, str | None]:\n"
        ),
    ),
    (
        "python_diverse_tests_scoreboard",
        (
            "# Implement the function so all assertions pass: rank players by score and name\n"
            "rows = [('Mia', 5), ('Ola', 8), ('Mia', 4), ('Ari', 9)]\n"
            "assert scoreboard(rows) == [('Ari', 9), ('Mia', 9), ('Ola', 8)]\n"
            "\n"
            "def scoreboard(events: list[tuple[str, int]]) -> list[tuple[str, int]]:\n"
        ),
    ),
    (
        "python_diverse_tests_normalize_headers",
        (
            "# Implement the function so all assertions pass: normalize table headers into unique snake_case fields\n"
            "assert normalize_headers(['First Name', 'First-Name', 'Age']) == ['first_name', 'first_name_2', 'age']\n"
            "assert normalize_headers(['  Total $ ']) == ['total']\n"
            "\n"
            "def normalize_headers(headers: list[str]) -> list[str]:\n"
        ),
    ),
    (
        "python_diverse_tests_histogram_bins",
        (
            "# Implement the function so all assertions pass: bucket values into fixed-width histogram bins\n"
            "assert histogram_bins([0, 1, 2, 5, 9], 5) == {'0-4': 3, '5-9': 2}\n"
            "assert histogram_bins([], 10) == {}\n"
            "\n"
            "def histogram_bins(values: list[int], width: int) -> dict[str, int]:\n"
        ),
    ),
    (
        "python_diverse_tests_route_params",
        (
            "# Implement the function so all assertions pass: match URL paths against route patterns\n"
            "assert route_params('/users/{id}/posts/{slug}', '/users/42/posts/hello') == {'id': '42', 'slug': 'hello'}\n"
            "assert route_params('/users/{id}', '/teams/42') is None\n"
            "\n"
            "def route_params(pattern: str, path: str) -> dict[str, str] | None:\n"
        ),
    ),
    (
        "python_diverse_dataclass_order_book",
        (
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class Order:\n"
            "    side: str\n"
            "    price: int\n"
            "    quantity: int\n"
            "    timestamp: int\n"
            "\n"
            "# Complete match_orders(buys, sells) -> list[tuple[int, int, int]].\n"
            "# Match highest buy prices with lowest sell prices while buy.price >= sell.price.\n"
            "# Earlier timestamps win within the same price level.\n"
            "# Each trade tuple is (trade_price, quantity, buy_timestamp).\n"
            "def match_orders(buys: list[Order], sells: list[Order]) -> list[tuple[int, int, int]]:\n"
        ),
    ),
    (
        "python_diverse_generator_batches",
        (
            "from collections.abc import Iterable, Iterator\n"
            "\n"
            "# Write a lazy generator that yields batches from any iterable.\n"
            "# The final batch may be shorter than size.\n"
            "# Raise ValueError when size is less than 1.\n"
            "def batched(iterable: Iterable[str], size: int) -> Iterator[list[str]]:\n"
        ),
    ),
    (
        "python_diverse_context_timer",
        (
            "import time\n"
            "\n"
            "class Timer:\n"
            "    \"\"\"Context manager that records elapsed seconds on exit.\"\"\"\n"
            "    # Implement __enter__ and __exit__.\n"
            "    # After the with block, instance.elapsed should contain a float duration.\n"
            "    # Exceptions from the with block must not be suppressed.\n"
        ),
    ),
    (
        "python_diverse_decorator_memoize",
        (
            "from functools import wraps\n"
            "\n"
            "# Complete a memoize decorator for functions called with hashable args and kwargs.\n"
            "# Preserve the wrapped function name and docstring.\n"
            "# Store cache entries by both positional args and sorted keyword args.\n"
            "def memoize(func):\n"
        ),
    ),
    (
        "python_diverse_async_gather_limited",
        (
            "import asyncio\n"
            "from collections.abc import Awaitable\n"
            "\n"
            "# Complete gather_limited so at most limit awaitables are running at once.\n"
            "# Return results in the same order as the input awaitables.\n"
            "# Let exceptions propagate like asyncio.gather.\n"
            "async def gather_limited(awaitables: list[Awaitable], limit: int) -> list[object]:\n"
        ),
    ),
    (
        "python_diverse_dataclass_tree",
        (
            "from dataclasses import dataclass, field\n"
            "\n"
            "@dataclass\n"
            "class Node:\n"
            "    name: str\n"
            "    children: list['Node'] = field(default_factory=list)\n"
            "\n"
            "# Complete find_path(root, target) to return node names from root to target.\n"
            "# Return None if target is not found. Use depth-first search.\n"
            "def find_path(root: Node, target: str) -> list[str] | None:\n"
        ),
    ),
    (
        "python_diverse_cli_flags",
        (
            "# Parse command-line style flags without using argparse.\n"
            "# Input example: ['--name', 'Ada', '--verbose', '--count=3']\n"
            "# Boolean flags have value True. Missing values after --key should also be True.\n"
            "# Return positional arguments under key '_'.\n"
            "def parse_flags(argv: list[str]) -> dict[str, object]:\n"
        ),
    ),
    (
        "python_diverse_sql_where_builder",
        (
            "# Build a SQL WHERE clause from filters safely.\n"
            "# Return a tuple of (clause, params).\n"
            "# Supported operators: eq, ne, gt, lt, in.\n"
            "# Example filter: {'age': ('gt', 18), 'status': ('in', ['new', 'open'])}\n"
            "# Use '?' placeholders and join clauses with AND. Sort field names for stability.\n"
            "def build_where(filters: dict[str, tuple[str, object]]) -> tuple[str, list[object]]:\n"
        ),
    ),
    (
        "python_diverse_event_sourcing",
        (
            "# Fold account events into current state.\n"
            "# Events are dictionaries with type values: opened, deposited, withdrawn, frozen.\n"
            "# Track balance, is_open, is_frozen, and rejected event ids.\n"
            "# Reject withdrawals that exceed balance or occur while frozen.\n"
            "def fold_account_events(events: list[dict[str, object]]) -> dict[str, object]:\n"
        ),
    ),
    (
        "python_diverse_mini_template",
        (
            "# Render a tiny template language.\n"
            "# Replace {{ name }} with values from context.\n"
            "# Support dotted lookup like {{ user.name }} through nested dicts.\n"
            "# Missing values should render as an empty string.\n"
            "def render_template(template: str, context: dict[str, object]) -> str:\n"
        ),
    ),
    (
        "python_diverse_retry_plan",
        (
            "# Generate retry delays with exponential backoff and jitter.\n"
            "# base is the first delay, factor multiplies each later delay, cap is the maximum.\n"
            "# jitter is a list of offsets to add cyclically after capping.\n"
            "# Return exactly attempts delays rounded to 3 decimals.\n"
            "def retry_delays(attempts: int, base: float, factor: float, cap: float, jitter: list[float]) -> list[float]:\n"
        ),
    ),
    (
        "python_diverse_dependency_explain",
        (
            "# Explain why a package is included in a dependency closure.\n"
            "# deps maps package -> direct dependencies.\n"
            "# Return one path from root to target as a list, or None if target is not reachable.\n"
            "# Prefer lexicographically smaller dependency names when several paths exist.\n"
            "def dependency_path(deps: dict[str, list[str]], root: str, target: str) -> list[str] | None:\n"
        ),
    ),
    (
        "python_diverse_streaming_median",
        (
            "import heapq\n"
            "\n"
            "class StreamingMedian:\n"
            "    # Implement add(value: int) and median() -> float | None.\n"
            "    # Use two heaps so each insertion is logarithmic.\n"
            "    # median returns None before any values are added.\n"
        ),
    ),
    (
        "python_diverse_file_tree",
        (
            "# Convert flat file paths into a nested dictionary tree.\n"
            "# Example: ['src/app.py', 'README.md'] -> {'src': {'app.py': None}, 'README.md': None}\n"
            "# Directories are dictionaries and files are None.\n"
            "# Ignore empty path components.\n"
            "def file_tree(paths: list[str]) -> dict[str, object]:\n"
        ),
    ),
    (
        "python_diverse_schema_validator",
        (
            "# Validate a dict against a small schema language.\n"
            "# Schema values are strings: 'str', 'int', 'float', 'bool', or 'list[str]'.\n"
            "# Return a list of human-readable error strings sorted by field name.\n"
            "# Missing fields and wrong types should both be reported.\n"
            "def validate_schema(data: dict[str, object], schema: dict[str, str]) -> list[str]:\n"
        ),
    ),
    (
        "python_diverse_interval_index",
        (
            "# Build an interval lookup function.\n"
            "# intervals are (start, end, label) with inclusive endpoints.\n"
            "# Return a function lookup(x) that returns all labels whose interval contains x.\n"
            "# Labels should be returned in the original interval order.\n"
            "def make_interval_lookup(intervals: list[tuple[int, int, str]]):\n"
        ),
    ),
    (
        "python_diverse_fuzzy_match",
        (
            "# Implement fuzzy subsequence matching for command names.\n"
            "# A command matches query if every query character appears in order, ignoring case.\n"
            "# Return matching commands sorted by fewest skipped characters, then alphabetically.\n"
            "def fuzzy_commands(commands: list[str], query: str) -> list[str]:\n"
        ),
    ),
    (
        "python_diverse_patch_dict",
        (
            "# Apply JSON-Patch-like operations to a dictionary.\n"
            "# Supported ops: add, replace, remove.\n"
            "# Paths are slash-separated keys such as /user/name. Support nested dicts only.\n"
            "# Return a new patched dictionary without mutating the input.\n"
            "def patch_dict(data: dict[str, object], ops: list[dict[str, object]]) -> dict[str, object]:\n"
        ),
    ),
    (
        "python_diverse_text_wrap",
        (
            "# Wrap text to lines of at most width characters.\n"
            "# Preserve words, collapse whitespace, and do not include trailing spaces.\n"
            "# If a word is longer than width, put it on its own line.\n"
            "def wrap_text(text: str, width: int) -> list[str]:\n"
        ),
    ),
    (
        "python_diverse_dag_schedule",
        (
            "# Schedule tasks with durations and dependencies.\n"
            "# durations maps task -> positive integer duration.\n"
            "# deps maps task -> prerequisites.\n"
            "# Return earliest finish time for each task, or raise ValueError on cycles.\n"
            "def earliest_finish_times(durations: dict[str, int], deps: dict[str, set[str]]) -> dict[str, int]:\n"
        ),
    ),
    (
        "python_diverse_sanitize_filename",
        (
            "# Sanitize a user-provided filename for cross-platform use.\n"
            "# Replace characters <>:\"/\\\\|?* with underscores, trim spaces and dots,\n"
            "# collapse repeated underscores, and return 'untitled' if nothing remains.\n"
            "def sanitize_filename(name: str) -> str:\n"
        ),
    ),
    (
        "python_diverse_merge_patch",
        (
            "# Implement JSON Merge Patch semantics for dictionaries.\n"
            "# A patch value of None removes a key.\n"
            "# Dict values are merged recursively; other values replace the target.\n"
            "# Return a new object without mutating target or patch.\n"
            "def merge_patch(target: dict[str, object], patch: dict[str, object]) -> dict[str, object]:\n"
        ),
    ),
    (
        "python_diverse_sparse_vector",
        (
            "# Complete sparse vector helpers.\n"
            "# Vectors are dict[int, float] and missing entries are zero.\n"
            "# add_sparse returns a new sparse vector without explicit zero entries.\n"
            "# cosine_similarity returns 0.0 if either vector has zero norm.\n"
            "def add_sparse(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:\n"
        ),
    ),
    (
        "python_diverse_tree_serialization",
        (
            "# Deserialize a binary tree encoded in level order with 'null' markers.\n"
            "# Return nested tuples of (value, left, right), where empty children are None.\n"
            "# Input tokens are strings; non-null values should remain strings.\n"
            "def deserialize_tree(tokens: list[str]) -> tuple[str, object, object] | None:\n"
        ),
    ),
    (
        "python_diverse_limited_map",
        (
            "# Map a function over values while collecting errors.\n"
            "# Return (results, errors), preserving input order in both lists.\n"
            "# If func raises, append {'index': i, 'error': str(exc)} to errors.\n"
            "def map_collect(func, values: list[object]) -> tuple[list[object], list[dict[str, object]]]:\n"
        ),
    ),
    (
        "python_diverse_config_env",
        (
            "# Overlay environment-style variables onto a nested config dictionary.\n"
            "# Keys use double underscores for nesting, e.g. APP__DB__HOST.\n"
            "# Strip the prefix, lowercase keys, and return a new config.\n"
            "def apply_env(config: dict[str, object], env: dict[str, str], prefix: str) -> dict[str, object]:\n"
        ),
    ),
    (
        "python_diverse_domino_chain",
        (
            "# Given dominoes as (left, right), build any chain using all dominoes once.\n"
            "# Adjacent dominoes must have matching touching values.\n"
            "# You may flip dominoes. Return None if no chain exists.\n"
            "def domino_chain(dominoes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:\n"
        ),
    ),
    (
        "python_diverse_table_join",
        (
            "# Perform an inner join of two in-memory tables.\n"
            "# Rows are dictionaries. Join on left_key and right_key.\n"
            "# Output rows merge both dictionaries, with right-side duplicate field names prefixed by 'right_'.\n"
            "# Preserve left row order, then right row order.\n"
            "def inner_join(left: list[dict], right: list[dict], left_key: str, right_key: str) -> list[dict]:\n"
        ),
    ),
    (
        "python_diverse_ascii_chart",
        (
            "# Render a horizontal ASCII bar chart.\n"
            "# Input rows are (label, value). Scale the largest value to width characters.\n"
            "# Align labels to the longest label and use '#' for bars.\n"
            "# Return a single string with newline-separated rows.\n"
            "def bar_chart(rows: list[tuple[str, int]], width: int) -> str:\n"
        ),
    ),
    (
        "python_diverse_reactive_cells",
        (
            "# Evaluate spreadsheet-like cells with formulas.\n"
            "# cells maps names to either numbers or formulas like '=A+B'.\n"
            "# Formulas may use + and references to other cells only.\n"
            "# Return resolved numeric values or raise ValueError on cycles.\n"
            "def evaluate_cells(cells: dict[str, object]) -> dict[str, float]:\n"
        ),
    ),
    (
        "python_diverse_multimap",
        (
            "# Implement a tiny MultiMap class.\n"
            "# add(key, value) appends a value. get(key) returns values in insertion order.\n"
            "# remove(key, value) removes one matching value and returns True, otherwise False.\n"
            "# items() yields (key, value) pairs grouped by key insertion order.\n"
            "class MultiMap:\n"
        ),
    ),
    (
        "python_diverse_url_canonicalize",
        (
            "# Canonicalize a simple URL without external libraries.\n"
            "# Lowercase scheme and host, remove default ports 80 for http and 443 for https,\n"
            "# sort query parameters by key then value, and remove a trailing slash from a non-root path.\n"
            "def canonical_url(url: str) -> str:\n"
        ),
    ),
    (
        "python_diverse_poll_aggregator",
        (
            "# Aggregate ranked-choice ballots using instant-runoff voting.\n"
            "# Each ballot is a list of candidate names in preference order.\n"
            "# Repeatedly eliminate the candidate with fewest active first-choice votes.\n"
            "# Break elimination ties alphabetically. Return the winner name.\n"
            "def instant_runoff(ballots: list[list[str]]) -> str | None:\n"
        ),
    ),
    (
        "python_diverse_pagination",
        (
            "# Paginate a list of items.\n"
            "# Return {'items': ..., 'page': page, 'pages': total_pages, 'total': total_count}.\n"
            "# Page numbers start at 1. Clamp page into the valid range when possible.\n"
            "# If per_page is less than 1, raise ValueError.\n"
            "def paginate(items: list[object], page: int, per_page: int) -> dict[str, object]:\n"
        ),
    ),
    (
        "python_diverse_trace_parser",
        (
            "# Parse a simple trace log into spans.\n"
            "# Lines look like '12 start request' or '20 end request'.\n"
            "# Return {name: duration}. Ignore incomplete spans.\n"
            "# If a name starts twice before ending, use the latest start.\n"
            "def parse_trace(lines: list[str]) -> dict[str, int]:\n"
        ),
    ),
    (
        "python_diverse_plugin_registry",
        (
            "# Implement a plugin registry decorator.\n"
            "# registry = make_registry() should return (register, get).\n"
            "# @register('name') stores the decorated function by name and returns the function unchanged.\n"
            "# get(name) returns the function or raises KeyError.\n"
            "def make_registry():\n"
        ),
    ),
    (
        "python_diverse_state_machine",
        (
            "# Validate transitions in a finite state machine.\n"
            "# transitions maps state -> allowed next states.\n"
            "# Return the final state after applying events, or raise ValueError for an invalid transition.\n"
            "def run_state_machine(start: str, events: list[str], transitions: dict[str, set[str]]) -> str:\n"
        ),
    ),
    (
        "python_diverse_autocomplete",
        (
            "# Build autocomplete suggestions from historical queries.\n"
            "# suggestions(prefix, limit) should return the most frequent queries with that prefix.\n"
            "# Sort by descending frequency, then alphabetically. Matching is case-insensitive.\n"
            "def make_autocomplete(queries: list[str]):\n"
        ),
    ),
    (
        "python_diverse_query_planner",
        (
            "# Choose an execution order for filters.\n"
            "# Each filter is {'name': str, 'cost': int, 'selectivity': float}.\n"
            "# Return filter names sorted by increasing cost * selectivity, then name.\n"
            "def plan_filters(filters: list[dict[str, object]]) -> list[str]:\n"
        ),
    ),
    (
        "python_diverse_audit_log",
        (
            "# Compress audit log events into human-readable summaries.\n"
            "# Consecutive events by the same user and action should be grouped.\n"
            "# Return strings like 'ada deleted 3 items'. Use singular 'item' for count 1.\n"
            "def summarize_audit(events: list[dict[str, str]]) -> list[str]:\n"
        ),
    ),
    (
        "python_diverse_feature_flags",
        (
            "# Resolve feature flags for a user.\n"
            "# flags maps flag name to rules with optional users, groups, percentage, and default fields.\n"
            "# Percentage rollout uses hash(user_id + flag_name) modulo 100.\n"
            "# Return a dict of flag name to boolean enabled value.\n"
            "def resolve_flags(flags: dict[str, dict], user_id: str, groups: set[str]) -> dict[str, bool]:\n"
        ),
    ),
    (
        "python_diverse_reorder_log",
        (
            "# Reorder log lines so letter-logs come before digit-logs.\n"
            "# A log line has an identifier followed by content. Letter-log content starts with a letter.\n"
            "# Sort letter-logs by content, then identifier. Keep digit-logs in original order.\n"
            "def reorder_logs(logs: list[str]) -> list[str]:\n"
        ),
    ),
    (
        "python_diverse_money_split",
        (
            "# Split an amount of cents among people as evenly as possible.\n"
            "# Return a dict name -> cents. Earlier names receive the extra cents when needed.\n"
            "# Raise ValueError if names is empty or amount is negative.\n"
            "def split_cents(amount: int, names: list[str]) -> dict[str, int]:\n"
        ),
    ),
    (
        "python_diverse_type_coerce",
        (
            "# Coerce string values according to a schema.\n"
            "# Schema values are 'int', 'float', 'bool', or 'str'.\n"
            "# Return (coerced, errors). Bool accepts true/false/yes/no/1/0 ignoring case.\n"
            "def coerce_row(row: dict[str, str], schema: dict[str, str]) -> tuple[dict[str, object], list[str]]:\n"
        ),
    ),
    (
        "python_diverse_matrix_regions",
        (
            "# Count connected regions of 1s in a grid of 0s and 1s.\n"
            "# Connections are four-directional only, not diagonal.\n"
            "# Do not mutate the input grid.\n"
            "def count_regions(grid: list[list[int]]) -> int:\n"
        ),
    ),
    (
        "python_diverse_cache_key",
        (
            "# Create a stable cache key for nested JSON-like data.\n"
            "# Dictionaries should be ordered by key, lists preserve order, and strings need repr-style quoting.\n"
            "# The same logical data should always produce the same key string.\n"
            "def stable_key(value: object) -> str:\n"
        ),
    ),
    (
        "python_diverse_number_words",
        (
            "# Convert an integer from 0 to 999 into English words.\n"
            "# Use lowercase words separated by spaces, e.g. 342 -> 'three hundred forty two'.\n"
            "# Raise ValueError outside the supported range.\n"
            "def number_to_words(n: int) -> str:\n"
        ),
    ),
    (
        "python_diverse_constraints_solver",
        (
            "# Solve a tiny exact-cover problem by backtracking.\n"
            "# options maps option name -> set of requirements it covers.\n"
            "# Return a list of option names covering every requirement exactly once, or None.\n"
            "# Prefer lexicographically smaller option names first.\n"
            "def exact_cover(requirements: set[str], options: dict[str, set[str]]) -> list[str] | None:\n"
        ),
    ),
    (
        "python_diverse_log_sampling",
        (
            "# Downsample noisy log events while preserving important records.\n"
            "# Always keep events whose level is ERROR or CRITICAL.\n"
            "# For other levels, keep at most one event per (level, message) per window seconds.\n"
            "# Return kept events in original order and do not mutate the inputs.\n"
            "def sample_logs(events: list[dict[str, object]], window: int) -> list[dict[str, object]]:\n"
        ),
    ),
    (
        "python_diverse_query_cache",
        (
            "# Complete a cache wrapper for expensive query functions.\n"
            "# The returned function should cache by normalized query string and sorted params.\n"
            "# Normalize queries by stripping leading/trailing whitespace and collapsing internal whitespace.\n"
            "# Expose cache_clear() on the returned function to empty the cache.\n"
            "def cached_query_runner(run_query):\n"
        ),
    ),
]

if len(INFERENCE_TEST_PROMPTS_PYTHON_DIVERSE) != 100:
    raise ValueError("Expected exactly 100 diverse Python prompts.")


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
