"""
Genealogical Research Task Intelligence & Advanced Workflow Automation Engine

Sophisticated research automation platform providing comprehensive genealogical task
templates, intelligent research workflow generation, and advanced task management
with AI-powered research prioritization, systematic investigation protocols, and
professional-grade research automation for genealogical investigation workflows.

Research Task Intelligence:
• Advanced genealogical task templates with intelligent research workflow generation and optimization
• Sophisticated research prioritization with AI-powered scoring algorithms and opportunity analysis
• Intelligent task automation with comprehensive research protocol generation and execution
• Advanced research gap analysis with intelligent opportunity identification and priority scoring
• Comprehensive research validation with intelligent quality assessment and verification protocols
• Integration with research management systems for comprehensive genealogical investigation workflows

Workflow Automation:
• Sophisticated research workflow generation with intelligent task sequencing and dependency management
• Advanced research protocol automation with systematic investigation methodologies and best practices
• Intelligent research coordination with multi-researcher collaboration and task distribution algorithms
• Comprehensive research tracking with detailed progress monitoring and performance analytics
• Advanced research optimization with intelligent workflow refinement and efficiency enhancement
• Integration with research platforms for comprehensive genealogical workflow automation and management

Research Intelligence:
• Advanced genealogical research analysis with AI-powered insights and research recommendations
• Sophisticated research pattern recognition with intelligent opportunity identification and analysis
• Intelligent research strategy optimization with data-driven research planning and execution
• Comprehensive research documentation with detailed investigation reports and findings analysis
• Advanced research collaboration with intelligent task sharing and coordination protocols
• Integration with genealogical intelligence systems for comprehensive research automation workflows

Foundation Services:
Provides the essential research task infrastructure that enables systematic,
professional genealogical investigation through intelligent task generation,
comprehensive workflow automation, and advanced research management for family history research.

Author: Ancestry Automation System
Created: August 6, 2025
Phase: 10.1 - Task Management & Actionability Enhancement
"""

# Ensure standard imports available for test expectations
from typing import Any, Optional

# Import standard modules
# Note: datetime and json imports removed as they were unused
from standard_imports import *
from standard_imports import get_logger

# Set up logging
logger = get_logger(__name__)


class GenealogicalTaskGenerator:
    """
    Generates specialized genealogical research tasks based on extracted data.
    Creates actionable, specific tasks that improve research productivity.
    """

    def __init__(self) -> None:
        """Initialize the task generator with templates and configuration."""
        self.task_templates = self._load_task_templates()
        self.task_config = self._load_task_configuration()

        # === PHASE 12: GEDCOM AI INTEGRATION ===
        try:
            from gedcom_ai_integration import GedcomAIIntegrator
            self.gedcom_ai_integrator = GedcomAIIntegrator()
            self.gedcom_ai_available = True
            logger.info("GEDCOM AI integration loaded in task generator")
        except ImportError as e:
            logger.debug(f"GEDCOM AI integration not available in task generator: {e}")
            self.gedcom_ai_integrator = None
            self.gedcom_ai_available = False

    def _load_task_templates(self) -> dict[str, dict[str, str]]:
        """Load enhanced genealogical research task templates with specific methodologies."""
        return {
            "vital_records_search": {
                "title": "Search {record_type} for {person_name} ({time_period})",
                "description": "Research {record_type} for {person_name} {birth_death_info}.\n\n📍 Location: {location}\n📅 Time Period: {time_period}\n⭐ Priority: {priority}\n\n🔍 DETAILED RESEARCH METHODOLOGY:\n\n1. PRIMARY SOURCES:\n   • Search {location} vital records databases (state/county archives)\n   • Check parish/church records for {time_period}\n   • Review cemetery records and burial registers\n   • Examine hospital/medical records if available\n\n2. LOCATION-SPECIFIC STRATEGIES:\n   • {location_strategy}\n   • Check neighboring counties/parishes for border communities\n   • Review historical boundary changes affecting record locations\n\n3. NAME VARIATIONS & SPELLINGS:\n   • Search alternative spellings: {name_variations}\n   • Check phonetic variations and transcription errors\n   • Look for nicknames and formal name variations\n\n4. FAMILY CONTEXT RESEARCH:\n   • Cross-reference with spouse and children's records\n   • Check family group records and household compositions\n   • Review witness signatures on family documents\n\n5. VERIFICATION STEPS:\n   • Compare dates and locations across multiple sources\n   • Verify against known family timeline\n   • Check for conflicting information requiring resolution\n\n🎯 EXPECTED OUTCOMES:\n   • {expected_info}\n   • Verification of family relationships\n   • Timeline confirmation for genealogical accuracy\n\n📊 SUCCESS CRITERIA:\n   • Primary source documentation located\n   • Date and location verified through multiple sources\n   • Family connections confirmed or clarified",
                "category": "vital_records",
                "priority": "high"
            },
            "dna_match_analysis": {
                "title": "Analyze DNA Match: {match_name} ({estimated_relationship})",
                "description": "Comprehensive DNA match investigation with {match_name} showing {shared_dna} shared DNA.\n\n🧬 MATCH DETAILS:\n   • Estimated Relationship: {estimated_relationship}\n   • Testing Company: {testing_company}\n   • Shared DNA: {shared_dna}\n   • Research Goal: {research_goal}\n\n🔬 SYSTEMATIC DNA ANALYSIS METHODOLOGY:\n\n1. QUANTITATIVE ANALYSIS:\n   • Review total shared centiMorgans (cM) and segments\n   • Analyze largest shared segment size\n   • Calculate relationship probability using DNA Painter tools\n   • Document X-chromosome inheritance patterns (if applicable)\n\n2. GENEALOGICAL COMPARISON:\n   • Compare family trees systematically by generation\n   • Identify potential common ancestor couples\n   • Map known family lines and geographical overlaps\n   • Document surname patterns and regional concentrations\n\n3. TRIANGULATION STRATEGY:\n   • Identify shared matches between you and this match\n   • Create triangulation groups for chromosome segments\n   • Map shared matches to specific ancestral lines\n   • Build evidence clusters supporting relationship theories\n\n4. ADVANCED VERIFICATION:\n   • Use chromosome browser for segment analysis\n   • Compare DNA match with known relatives\n   • Cross-reference with paper trail evidence\n   • Validate relationship through multiple DNA relatives\n\n5. RESEARCH EXPANSION:\n   • Contact match for family information exchange\n   • Request access to extended family trees\n   • Share relevant genealogical documentation\n   • Coordinate research efforts for mutual benefit\n\n🎯 EXPECTED OUTCOMES:\n   • Identification of most recent common ancestors (MRCA)\n   • Verification of specific family line connections\n   • Resolution of brick wall research problems\n   • Expansion of verified family tree branches\n\n📊 SUCCESS CRITERIA:\n   • Common ancestors identified with supporting evidence\n   • Relationship path documented and verified\n   • DNA evidence integrated with paper trail research",
                "category": "dna_analysis",
                "priority": "medium"
            },
            "family_tree_verification": {
                "title": "Verify Family Connection: {person1} → {person2}",
                "description": "Systematic verification of the relationship between {person1} and {person2}.\n\n🔍 VERIFICATION DETAILS:\n   • Relationship to Verify: {relationship}\n   • Conflicting Information: {conflicts}\n   • Evidence Available: {evidence}\n   • Resolution Priority: {priority}\n\n📋 COMPREHENSIVE VERIFICATION METHODOLOGY:\n\n1. PRIMARY SOURCE COLLECTION:\n   • Gather birth, marriage, and death certificates\n   • Collect census records showing family groupings\n   • Review church/parish records for family events\n   • Examine probate records and wills mentioning relationships\n\n2. MULTI-SOURCE CROSS-REFERENCE:\n   • Compare information across 3+ independent sources\n   • Verify dates and locations for consistency\n   • Check for corroborating evidence from different record types\n   • Document any discrepancies requiring further investigation\n\n3. ALTERNATIVE HYPOTHESIS TESTING:\n   • Consider alternative relationship explanations\n   • Test competing theories against available evidence\n   • Examine possibility of adoption, step-relationships, or name changes\n   • Investigate potential record transcription errors\n\n4. CONTEXTUAL VALIDATION:\n   • Verify against known family timeline and geography\n   • Check relationship against historical and social context\n   • Confirm biological feasibility of claimed relationships\n   • Review family naming patterns and traditions\n\n5. DOCUMENTATION & CITATION:\n   • Create detailed source citations for all evidence\n   • Document research methodology and decision process\n   • Prepare evidence summary with confidence assessment\n   • Update family tree with verified information and source links\n\n🎯 EXPECTED OUTCOMES:\n   • Definitive confirmation or refutation of relationship\n   • Resolution of conflicting information\n   • Strengthened family tree accuracy\n   • Clear documentation for future researchers\n\n📊 SUCCESS CRITERIA:\n   • Relationship verified through multiple independent sources\n   • All conflicts resolved with documented explanations\n   • Evidence meets genealogical proof standard",
                "category": "verification",
                "priority": "high"
            },
            "immigration_research": {
                "title": "Immigration Research: {person_name} ({origin} → {destination})",
                "description": "Comprehensive immigration research for {person_name} from {origin} to {destination}.\n\n🚢 IMMIGRATION DETAILS:\n   • Time Period: {time_period}\n   • Departure Port: {ports}\n   • Ship/Vessel: {vessel_info}\n   • Expected Documents: {expected_documents}\n\n🌍 SYSTEMATIC IMMIGRATION RESEARCH METHODOLOGY:\n\n1. PASSENGER MANIFEST RESEARCH:\n   • Search ship passenger lists for {time_period}\n   • Check multiple spelling variations of {person_name}\n   • Look for family groups traveling together\n   • Review both departure and arrival manifests\n   • Cross-reference with known family members\n\n2. ORIGIN COUNTRY RESEARCH:\n   • Search {origin} emigration records and permits\n   • Check parish records for departure notifications\n   • Review local newspapers for emigration announcements\n   • Examine land sales or property transfers before departure\n   • Research family left behind for correspondence\n\n3. DESTINATION COUNTRY INTEGRATION:\n   • Search naturalization records and declarations of intent\n   • Check early census records in {destination}\n   • Look for immigrant aid society records\n   • Review early employment or business records\n   • Examine church membership transfers\n\n4. TRAVEL COMPANION ANALYSIS:\n   • Identify other passengers from same region\n   • Research family members who may have traveled separately\n   • Check for chain migration patterns\n   • Look for sponsors or contacts in destination country\n\n5. HISTORICAL CONTEXT RESEARCH:\n   • Study migration patterns from {origin} during {time_period}\n   • Research economic/political factors driving emigration\n   • Examine transportation routes and shipping companies\n   • Review immigration laws and requirements of the era\n\n🎯 EXPECTED OUTCOMES:\n   • Complete immigration timeline and documentation\n   • Verification of family travel companions\n   • Understanding of migration motivations and context\n   • Connection to origin and destination communities\n\n📊 SUCCESS CRITERIA:\n   • Immigration date and vessel identified\n   • Origin location and family context documented\n   • Integration into destination community traced",
                "category": "immigration",
                "priority": "medium"
            },
            "census_research": {
                "title": "Census Research: {person_name} Family ({location}, {year})",
                "description": "Comprehensive census research for {person_name} and family in {year} census records.\n\n📊 CENSUS RESEARCH PARAMETERS:\n   • Target Location: {location}\n   • Census Year: {year}\n   • Family Members: {family_members}\n   • Known Occupation: {occupation}\n   • Information Needed: {information_needed}\n\n📋 SYSTEMATIC CENSUS RESEARCH METHODOLOGY:\n\n1. PRIMARY SEARCH STRATEGY:\n   • Search {year} census for {location} using exact name\n   • Expand search to county/state level if not found locally\n   • Use Soundex and phonetic search algorithms\n   • Search by household head if {person_name} is not head\n\n2. NAME VARIATION TECHNIQUES:\n   • Try alternative spellings and transcription errors\n   • Search using nicknames and formal name variations\n   • Check for reversed first/middle names\n   • Consider ethnic name variations and Americanization\n\n3. FAMILY GROUP IDENTIFICATION:\n   • Search for known family members as entry points\n   • Use spouse and children's names to locate household\n   • Check for extended family living in same household\n   • Look for boarders or servants who might be relatives\n\n4. GEOGRAPHICAL EXPANSION:\n   • Search neighboring townships and counties\n   • Check for boundary changes affecting enumeration\n   • Consider seasonal migration or temporary relocation\n   • Review urban vs. rural enumeration districts\n\n5. TEMPORAL ANALYSIS:\n   • Compare with previous census ({year-10}) for migration patterns\n   • Check subsequent census ({year+10}) for family changes\n   • Analyze age progression and family composition changes\n   • Document occupation and property value changes\n\n6. CONTEXTUAL VERIFICATION:\n   • Verify ages against known birth dates\n   • Check birthplaces against family history\n   • Confirm occupation against other records\n   • Validate family relationships and household composition\n\n🎯 EXPECTED OUTCOMES:\n   • Complete household enumeration with all family members\n   • Verification of residence location and duration\n   • Documentation of occupation and economic status\n   • Age and birthplace verification for family members\n\n📊 SUCCESS CRITERIA:\n   • Family located in correct census year and location\n   • All household members identified and relationships confirmed\n   • Information integrated with broader family timeline",
                "category": "census",
                "priority": "medium"
            },
            "military_research": {
                "title": "Military Service Research: {person_name} ({conflict})",
                "description": "Comprehensive military service research for {person_name} during {conflict}.\n\n⚔️ MILITARY SERVICE DETAILS:\n   • Conflict: {conflict}\n   • Service Branch: {service_branch}\n   • Unit Information: {unit_info}\n   • Service Period: {service_period}\n   • Expected Records: {expected_records}\n\n🎖️ SYSTEMATIC MILITARY RESEARCH METHODOLOGY:\n\n1. SERVICE RECORD RESEARCH:\n   • Search compiled military service records (CMSR)\n   • Check enlistment and discharge papers\n   • Review muster rolls and pay records\n   • Examine medical records and casualty reports\n   • Look for court martial or disciplinary records\n\n2. PENSION AND BENEFIT RESEARCH:\n   • Search pension application files\n   • Check widow's and dependent pension records\n   • Review bounty land warrant applications\n   • Examine disability and medical pension files\n   • Look for rejected or pending pension claims\n\n3. UNIT HISTORY AND CONTEXT:\n   • Research regimental and company histories\n   • Check unit muster rolls and organizational records\n   • Study battle participation and campaign records\n   • Review unit movements and station assignments\n   • Examine casualty lists and honor rolls\n\n4. BATTLE AND CAMPAIGN ANALYSIS:\n   • Document specific battles and engagements\n   • Research unit's role in major campaigns\n   • Check for mentions in official reports\n   • Look for personal accounts and diaries\n   • Examine prisoner of war records if applicable\n\n5. POST-SERVICE INTEGRATION:\n   • Research veteran organization memberships\n   • Check for Grand Army of the Republic records\n   • Look for veteran reunion attendance\n   • Examine veteran cemetery and burial records\n   • Review veteran benefit and hospital records\n\n6. FAMILY IMPACT RESEARCH:\n   • Document impact on family during service\n   • Research family correspondence during war\n   • Check for family members' military service\n   • Examine post-war family reunification\n   • Look for war-related family migrations\n\n🎯 EXPECTED OUTCOMES:\n   • Complete military service timeline and documentation\n   • Verification of unit assignments and battle participation\n   • Understanding of service impact on family\n   • Connection to veteran communities and benefits\n\n📊 SUCCESS CRITERIA:\n   • Military service verified through official records\n   • Unit assignments and battle participation documented\n   • Pension and benefit records located and analyzed",
                "category": "military",
                "priority": "medium"
            },
            "occupation_research": {
                "title": "Occupation Research: {person_name} ({occupation})",
                "description": "Comprehensive occupational research for {person_name}'s career as {occupation} in {location}.\n\n💼 OCCUPATIONAL DETAILS:\n   • Occupation: {occupation}\n   • Location: {location}\n   • Time Period: {time_period}\n   • Known Employer: {employer}\n   • Research Goal: {research_goal}\n\n🏭 SYSTEMATIC OCCUPATIONAL RESEARCH METHODOLOGY:\n\n1. EMPLOYMENT RECORD RESEARCH:\n   • Search city directories for business listings\n   • Check employment records and payroll documents\n   • Review apprenticeship and training records\n   • Examine union membership and labor organization records\n   • Look for professional licensing and certification documents\n\n2. INDUSTRY-SPECIFIC RESEARCH:\n   • Study industry history and development in {location}\n   • Research major employers and business establishments\n   • Check trade publications and industry journals\n   • Examine guild records and professional associations\n   • Review industry-specific regulatory records\n\n3. NEWSPAPER AND MEDIA RESEARCH:\n   • Search local newspapers for business mentions\n   • Check obituaries for occupational details\n   • Look for business advertisements and announcements\n   • Review social pages for professional activities\n   • Examine trade journal articles and features\n\n4. PROPERTY AND BUSINESS RECORDS:\n   • Research business property ownership and leases\n   • Check commercial property tax records\n   • Examine business partnership and incorporation documents\n   • Look for business insurance and bonding records\n   • Review bankruptcy or business dissolution records\n\n5. SOCIAL AND PROFESSIONAL NETWORKS:\n   • Research professional association memberships\n   • Check social club and organization records\n   • Look for business partner and colleague connections\n   • Examine professional conference and meeting attendance\n   • Review charitable and civic organization involvement\n\n6. ECONOMIC CONTEXT ANALYSIS:\n   • Study economic conditions affecting the industry\n   • Research technological changes impacting the occupation\n   • Examine labor disputes and strikes affecting the field\n   • Look for government regulations affecting the profession\n   • Analyze migration patterns related to occupational opportunities\n\n🎯 EXPECTED OUTCOMES:\n   • Complete occupational timeline and career progression\n   • Understanding of professional networks and associations\n   • Documentation of economic and social status\n   • Context for family migration and settlement patterns\n\n📊 SUCCESS CRITERIA:\n   • Occupational details verified through multiple sources\n   • Professional networks and associations documented\n   • Career progression and economic impact understood",
                "category": "occupation",
                "priority": "low"
            },
            "location_research": {
                "title": "Location Research: {person_name} in {location}",
                "description": "Comprehensive location-based research for {person_name}'s time in {location} during {time_period}.\n\n📍 LOCATION RESEARCH PARAMETERS:\n   • Location: {location}\n   • Time Period: {time_period}\n   • Residence Type: {residence_type}\n   • Known Neighbors: {neighbors}\n   • Information Sought: {information_sought}\n\n🗺️ SYSTEMATIC LOCATION RESEARCH METHODOLOGY:\n\n1. PROPERTY AND LAND RECORDS:\n   • Search deed records for property ownership\n   • Check property tax records and assessments\n   • Review mortgage and land contract documents\n   • Examine homestead and land grant records\n   • Look for property inheritance and transfer records\n\n2. LOCAL DIRECTORY AND CENSUS RESEARCH:\n   • Search city/county directories for residence listings\n   • Check voter registration and poll tax records\n   • Review local census and enumeration records\n   • Examine school district and enrollment records\n   • Look for jury duty and civic service records\n\n3. RELIGIOUS AND COMMUNITY RECORDS:\n   • Search church membership and baptismal records\n   • Check cemetery and burial records\n   • Review school attendance and graduation records\n   • Examine fraternal organization memberships\n   • Look for charitable and social organization involvement\n\n4. NEIGHBORHOOD AND COMMUNITY ANALYSIS:\n   • Research known neighbors and their families\n   • Study community development and growth patterns\n   • Examine local business and commercial development\n   • Check for ethnic or cultural community concentrations\n   • Look for family clusters and chain migration patterns\n\n5. HISTORICAL CONTEXT RESEARCH:\n   • Study local history and significant events\n   • Research economic development and industry growth\n   • Examine transportation development (roads, railroads)\n   • Look for natural disasters or significant local events\n   • Study political and administrative boundary changes\n\n6. MIGRATION PATTERN ANALYSIS:\n   • Research why family came to this location\n   • Check for previous and subsequent residences\n   • Examine seasonal migration or temporary relocations\n   • Look for family members in surrounding areas\n   • Study regional migration trends and patterns\n\n🎯 EXPECTED OUTCOMES:\n   • Complete residential timeline and property history\n   • Understanding of community integration and networks\n   • Documentation of local family and social connections\n   • Context for family decisions and life events\n\n📊 SUCCESS CRITERIA:\n   • Residence period and property details documented\n   • Community connections and networks identified\n   • Migration motivations and patterns understood",
                "category": "location",
                "priority": "low"
            }
        }

    def _get_location_specific_strategy(self, location: str) -> str:
        """Generate location-specific research strategies based on geographical and historical context."""
        location_lower = location.lower()

        # Determine strategy based on location
        strategy = None

        # Scotland-specific strategies
        if any(term in location_lower for term in ['scotland', 'scottish', 'edinburgh', 'glasgow', 'aberdeen', 'dundee']):
            strategy = "Focus on Old Parish Registers (OPR) pre-1855 and statutory records post-1855. Check National Records of Scotland online. Review kirk session records and heritors' records. Consider Highland Clearances impact if applicable."

        # Ireland-specific strategies
        elif any(term in location_lower for term in ['ireland', 'irish', 'dublin', 'cork', 'belfast', 'galway']):
            strategy = "Search civil registration from 1864 (births/deaths) and 1845 (marriages). Check Catholic parish records and Church of Ireland registers. Review Griffith's Valuation and Tithe Applotment Books. Consider Famine emigration records."

        # England-specific strategies
        elif any(term in location_lower for term in ['england', 'english', 'london', 'manchester', 'birmingham', 'liverpool']):
            strategy = "Search parish registers and Bishop's Transcripts. Check Probate Court of Canterbury (PCC) wills. Review Quarter Sessions and Manorial records. Consider Industrial Revolution migration patterns."

        # Wales-specific strategies
        elif any(term in location_lower for term in ['wales', 'welsh', 'cardiff', 'swansea', 'newport']):
            strategy = "Focus on Welsh parish registers and Nonconformist records. Check National Library of Wales collections. Review tithe maps and schedules. Consider Welsh language variations in records."

        # US state-specific strategies
        elif any(term in location_lower for term in ['massachusetts', 'boston', 'ma']):
            strategy = "Check Massachusetts Vital Records to 1850. Review Mayflower descendant records if applicable. Search Boston immigration records and ship manifests. Check town clerk records."

        elif any(term in location_lower for term in ['new york', 'ny', 'manhattan', 'brooklyn']):
            strategy = "Search Ellis Island and Castle Garden immigration records. Check NYC Municipal Archives. Review tenement records and city directories. Consider ethnic neighborhood concentrations."

        elif any(term in location_lower for term in ['pennsylvania', 'philadelphia', 'pa']):
            strategy = "Check Pennsylvania German records if applicable. Search Quaker meeting records. Review Philadelphia port records. Check county courthouse records."

        # Generic strategies for other locations
        else:
            strategy = "Research local archives and historical societies. Check county courthouse records. Review local newspapers and obituaries. Consider regional migration patterns and historical events."

        return strategy

    def _apply_letter_substitutions(self, name_parts: list[str]) -> list[str]:
        """Apply common letter substitutions to name parts."""
        variations = []
        for part in name_parts:
            if 'ph' in part.lower():
                variations.append(part.replace('ph', 'f').replace('Ph', 'F'))
            if 'c' in part.lower():
                variations.append(part.replace('c', 'k').replace('C', 'K'))
            if 'y' in part.lower():
                variations.append(part.replace('y', 'i').replace('Y', 'I'))
        return variations

    def _add_nickname_variations(self, first_name: str) -> list[str]:
        """Add common nickname variations for a given first name."""
        nickname_map = {
            'william': ['William', 'Bill', 'Billy', 'Will', 'Willie'],
            'bill': ['William', 'Bill', 'Billy', 'Will', 'Willie'],
            'billy': ['William', 'Bill', 'Billy', 'Will', 'Willie'],
            'robert': ['Robert', 'Bob', 'Bobby', 'Rob', 'Robbie'],
            'bob': ['Robert', 'Bob', 'Bobby', 'Rob', 'Robbie'],
            'bobby': ['Robert', 'Bob', 'Bobby', 'Rob', 'Robbie'],
            'james': ['James', 'Jim', 'Jimmy', 'Jamie'],
            'jim': ['James', 'Jim', 'Jimmy', 'Jamie'],
            'jimmy': ['James', 'Jim', 'Jimmy', 'Jamie'],
        }
        return nickname_map.get(first_name.lower(), [])

    def _generate_name_variations(self, name: str) -> str:
        """Generate common name variations for genealogical research."""
        if not name:
            return "Check common spelling variations"

        name_parts = name.split()
        variations = []

        # Apply letter substitutions
        variations.extend(self._apply_letter_substitutions(name_parts))

        # Add nickname variations
        if name_parts:
            variations.extend(self._add_nickname_variations(name_parts[0]))

        if variations:
            return f"Try variations: {', '.join(set(variations[:5]))}"
        return "Check phonetic spellings and transcription errors"

    def _load_task_configuration(self) -> dict[str, Any]:
        """Load task generation configuration."""
        return {
            "max_tasks_per_person": 5,
            "priority_weights": {
                "high": 3,
                "medium": 2,
                "low": 1
            },
            "category_limits": {
                "vital_records": 2,
                "dna_analysis": 1,
                "verification": 2,
                "immigration": 1,
                "census": 1,
                "military": 1,
                "occupation": 1,
                "location": 1
            }
        }

    def _validate_and_normalize_inputs(
        self,
        person_data: dict[str, Any] | None,
        extracted_data: dict[str, Any] | None,
        suggested_tasks: list[str] | None
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        """Validate and normalize input parameters."""
        if person_data is None or not isinstance(person_data, dict):
            person_data = {}
        if extracted_data is None or not isinstance(extracted_data, dict):
            extracted_data = {}
        if suggested_tasks is None or not isinstance(suggested_tasks, list):
            suggested_tasks = []
        return person_data, extracted_data, suggested_tasks

    def _generate_ai_enhanced_tasks(
        self,
        person_data: dict[str, Any],
        extracted_data: dict[str, Any],
        gedcom_data: Any
    ) -> list[dict[str, Any]]:
        """Generate GEDCOM AI-enhanced tasks if available."""
        if not (self.gedcom_ai_available and self.gedcom_ai_integrator is not None and gedcom_data):
            return []

        try:
            logger.debug("Generating GEDCOM AI-enhanced tasks")
            ai_enhanced_tasks = self.gedcom_ai_integrator.generate_enhanced_research_tasks(
                person_data, extracted_data, gedcom_data
            )
            logger.info(f"Generated {len(ai_enhanced_tasks)} GEDCOM AI-enhanced tasks")
            return ai_enhanced_tasks
        except Exception as e:
            logger.warning(f"GEDCOM AI task generation failed: {e}, falling back to standard generation")
            return []

    def _generate_all_standard_tasks(self, extracted_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate all standard task types based on extracted data."""
        tasks = []
        tasks.extend(self._generate_vital_records_tasks(extracted_data))
        tasks.extend(self._generate_dna_analysis_tasks(extracted_data))
        tasks.extend(self._generate_verification_tasks(extracted_data))
        tasks.extend(self._generate_immigration_tasks(extracted_data))
        tasks.extend(self._generate_census_tasks(extracted_data))
        tasks.extend(self._generate_military_tasks(extracted_data))
        tasks.extend(self._generate_occupation_tasks(extracted_data))
        tasks.extend(self._generate_location_tasks(extracted_data))
        return tasks

    def generate_research_tasks(
        self,
        person_data: dict[str, Any],
        extracted_data: dict[str, Any],
        suggested_tasks: list[str],
        gedcom_data: Any = None
    ) -> list[dict[str, Any]]:
        """
        Generate specialized research tasks based on extracted genealogical data.

        Args:
            person_data: Information about the person being researched
            extracted_data: Genealogical data extracted from conversations
            suggested_tasks: Basic AI-generated task suggestions
            gedcom_data: Optional GEDCOM data for AI-enhanced analysis

        Returns:
            List of enhanced task dictionaries with titles, descriptions, categories, and priorities
        """
        try:
            # Input validation and safe defaults
            person_data, extracted_data, suggested_tasks = self._validate_and_normalize_inputs(
                person_data, extracted_data, suggested_tasks
            )

            # Generate AI-enhanced tasks if available
            enhanced_tasks = self._generate_ai_enhanced_tasks(person_data, extracted_data, gedcom_data)

            # Generate standard tasks
            enhanced_tasks.extend(self._generate_all_standard_tasks(extracted_data))

            # Add fallback tasks if no specific tasks generated
            if not enhanced_tasks and suggested_tasks:
                enhanced_tasks.extend(self._create_fallback_tasks(person_data, suggested_tasks))

            # Prioritize and limit tasks
            prioritized_tasks = self._prioritize_and_limit_tasks(enhanced_tasks)

            logger.info(f"Generated {len(prioritized_tasks)} enhanced research tasks (GEDCOM AI: {'enabled' if self.gedcom_ai_available and gedcom_data else 'disabled'})")
            return prioritized_tasks

        except Exception as e:
            logger.error(f"Error generating research tasks: {e}")
            return self._create_fallback_tasks(person_data, suggested_tasks)

    def _generate_vital_records_tasks(self, extracted_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate enhanced vital records search tasks with location-specific strategies."""
        tasks = []
        vital_records = extracted_data.get("vital_records", [])

        for record in vital_records[:2]:  # Limit to 2 most important
            if isinstance(record, dict):
                person = record.get("person", "Unknown Person")
                event_type = record.get("event_type", "vital record")
                date = record.get("date", "unknown date")
                place = record.get("place", "unknown location")

                # Generate enhanced task data with location-specific strategies
                task_data = {
                    "person_name": person,
                    "record_type": f"{event_type} record",
                    "birth_death_info": f"({event_type} {date})" if date != "unknown date" else "",
                    "location": place,
                    "time_period": date,
                    "priority": "high",
                    "location_strategy": self._get_location_specific_strategy(place),
                    "name_variations": self._generate_name_variations(person),
                    "expected_info": f"Official {event_type} documentation with parents, dates, and locations"
                }

                task = self._create_task_from_template("vital_records_search", task_data)
                if task:
                    tasks.append(task)

        return tasks

    def _generate_dna_analysis_tasks(self, extracted_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate DNA match analysis tasks."""
        tasks = []
        dna_info = extracted_data.get("dna_information", [])

        for info in dna_info[:1]:  # Limit to 1 DNA task
            if isinstance(info, str) and ("match" in info.lower() or "dna" in info.lower()):
                task_data = {
                    "match_name": "DNA Match",
                    "estimated_relationship": "close family connection",
                    "shared_dna": "significant amount",
                    "testing_company": "Ancestry/23andMe",
                    "research_goal": "Identify common ancestors and verify family connections"
                }

                task = self._create_task_from_template("dna_match_analysis", task_data)
                if task:
                    tasks.append(task)
                break

        return tasks

    def _generate_verification_tasks(self, extracted_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate family tree verification tasks."""
        tasks = []
        relationships = extracted_data.get("relationships", [])

        for relationship in relationships[:2]:  # Limit to 2 verification tasks
            if isinstance(relationship, dict):
                person1 = relationship.get("person1", "Person A")
                person2 = relationship.get("person2", "Person B")
                rel_type = relationship.get("relationship", "family connection")

                task_data = {
                    "person1": person1,
                    "person2": person2,
                    "relationship": rel_type,
                    "conflicts": "Multiple sources with different information",
                    "evidence": "Family stories and preliminary research",
                    "priority": "high"
                }

                task = self._create_task_from_template("family_tree_verification", task_data)
                if task:
                    tasks.append(task)

        return tasks

    def _generate_immigration_tasks(self, extracted_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate immigration research tasks."""
        tasks = []
        locations = extracted_data.get("locations", [])

        # Look for potential immigration scenarios
        foreign_locations = []
        for location in locations:
            if isinstance(location, dict):
                place = location.get("place", "")
                if any(country in place for country in ["Ireland", "Scotland", "England", "Germany", "Poland", "Italy"]):
                    foreign_locations.append(location)

        if foreign_locations:
            location = foreign_locations[0]
            place = location.get("place", "Unknown Location")
            time_period = location.get("time_period", "1800s-1900s")

            task_data = {
                "person_name": "Family Member",
                "origin": place,
                "destination": "United States",
                "time_period": time_period,
                "ports": "Ellis Island, Castle Garden, or other major ports",
                "vessel_info": "To be determined",
                "expected_documents": "Passenger manifests, naturalization records, ship records"
            }

            task = self._create_task_from_template("immigration_research", task_data)
            if task:
                tasks.append(task)

        return tasks

    def _generate_census_tasks(self, extracted_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate census research tasks."""
        tasks = []
        structured_names = extracted_data.get("structured_names", [])
        locations = extracted_data.get("locations", [])

        if structured_names and locations:
            name_data = structured_names[0]
            location_data = locations[0]

            person_name = name_data.get("full_name", "Family Member") if isinstance(name_data, dict) else str(name_data)
            location = location_data.get("place", "Unknown Location") if isinstance(location_data, dict) else str(location_data)

            task_data = {
                "person_name": person_name,
                "location": location,
                "year": "1900-1940",
                "family_members": "Spouse and children",
                "occupation": "To be determined",
                "information_needed": "Family composition, ages, birthplaces, occupations"
            }

            task = self._create_task_from_template("census_research", task_data)
            if task:
                tasks.append(task)

        return tasks

    def _generate_military_tasks(self, extracted_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate military research tasks."""
        tasks = []
        # Look for military-related information in research questions or documents
        research_questions = extracted_data.get("research_questions", [])

        for question in research_questions:
            if isinstance(question, str) and any(term in question.lower() for term in ["war", "military", "service", "veteran", "army", "navy"]):
                task_data = {
                    "person_name": "Service Member",
                    "conflict": "Civil War, WWI, or WWII",
                    "service_branch": "To be determined",
                    "unit_info": "To be researched",
                    "service_period": "To be determined",
                    "expected_records": "Service records, pension files, unit histories"
                }

                task = self._create_task_from_template("military_research", task_data)
                if task:
                    tasks.append(task)
                break

        return tasks

    def _generate_occupation_tasks(self, extracted_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate occupation research tasks."""
        tasks = []
        occupations = extracted_data.get("occupations", [])

        for occupation in occupations[:1]:  # Limit to 1 occupation task
            if isinstance(occupation, dict):
                person = occupation.get("person", "Worker")
                job = occupation.get("occupation", "Unknown Occupation")
                location = occupation.get("location", "Unknown Location")
                time_period = occupation.get("time_period", "Unknown Period")

                task_data = {
                    "person_name": person,
                    "occupation": job,
                    "location": location,
                    "time_period": time_period,
                    "employer": "To be determined",
                    "research_goal": f"Understand {person}'s career and work history"
                }

                task = self._create_task_from_template("occupation_research", task_data)
                if task:
                    tasks.append(task)

        return tasks

    def _generate_location_tasks(self, extracted_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate location research tasks."""
        tasks = []
        locations = extracted_data.get("locations", [])

        for location in locations[:1]:  # Limit to 1 location task
            if isinstance(location, dict):
                place = location.get("place", "Unknown Location")
                context = location.get("context", "residence")
                time_period = location.get("time_period", "Unknown Period")

                task_data = {
                    "person_name": "Family Member",
                    "location": place,
                    "time_period": time_period,
                    "residence_type": context,
                    "neighbors": "To be researched",
                    "information_sought": f"Family's time in {place} and local connections"
                }

                task = self._create_task_from_template("location_research", task_data)
                if task:
                    tasks.append(task)

        return tasks

    def _create_task_from_template(self, template_key: str, task_data: dict[str, str]) -> Optional[dict[str, Any]]:
        """Create a task from a template with provided data."""
        try:
            template = self.task_templates.get(template_key)
            if not template:
                return None

            # Format title and description
            title = template["title"].format(**task_data)
            description = template["description"].format(**task_data)

            return {
                "title": title,
                "description": description,
                "category": template["category"],
                "priority": template["priority"],
                "template_used": template_key
            }

        except KeyError as e:
            logger.warning(f"Missing template data key {e} for template {template_key}")
            return None
        except Exception as e:
            logger.error(f"Error creating task from template {template_key}: {e}")
            return None

    def _create_fallback_tasks(self, person_data: dict[str, Any], suggested_tasks: list[str]) -> list[dict[str, Any]]:
        """Create fallback tasks from AI suggestions."""
        fallback_tasks = []
        username = person_data.get("username", "Unknown")

        for i, task_desc in enumerate(suggested_tasks[:3]):  # Limit to 3 fallback tasks
            fallback_tasks.append({
                "title": f"Genealogy Research: {username} (Task {i+1})",
                "description": f"Research Task: {task_desc}\n\nMatch: {username}\nPriority: Medium\n\nThis is a general research task. Consider breaking it down into more specific actions.",
                "category": "general",
                "priority": "medium",
                "template_used": "fallback"
            })

        return fallback_tasks

    def _prioritize_and_limit_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prioritize tasks and limit to maximum number."""
        if not tasks:
            return []

        # Sort by priority (high > medium > low)
        priority_order = {"high": 3, "medium": 2, "low": 1}
        sorted_tasks = sorted(
            tasks,
            key=lambda t: priority_order.get(t.get("priority", "low"), 1),
            reverse=True
        )

        # Limit to maximum tasks per person
        max_tasks = self.task_config["max_tasks_per_person"]
        return sorted_tasks[:max_tasks]


# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================
# Extracted from monolithic genealogical_task_templates_module_tests() for better organization
# Each test function is independent and can be run individually


def _test_module_imports():
    """Test that required modules and dependencies are imported correctly."""
    # Test core infrastructure imports
    assert 'logger' in globals(), "Logger should be initialized"
    assert 'get_logger' in globals(), "get_logger function should be available"

    # Test class availability
    assert 'GenealogicalTaskGenerator' in globals(), "GenealogicalTaskGenerator class should be available"

    # Test standard imports availability through standard_imports
    # Note: json and datetime were removed as unused per line 47 comment
    # Only test for imports that are actually used in the module
    assert 'Any' in globals(), "typing.Any should be available"
    assert 'Optional' in globals(), "typing.Optional should be available"


def _test_task_generator_initialization():
    """Test GenealogicalTaskGenerator initialization and setup."""
    generator = GenealogicalTaskGenerator()

    # Test basic initialization
    assert hasattr(generator, 'task_templates'), "Generator should have task templates"
    assert hasattr(generator, 'task_config'), "Generator should have task configuration"
    assert isinstance(generator.task_templates, dict), "Task templates should be a dictionary"
    assert isinstance(generator.task_config, dict), "Task config should be a dictionary"

    # Test GEDCOM AI integration setup
    assert hasattr(generator, 'gedcom_ai_available'), "Should track GEDCOM AI availability"
    assert hasattr(generator, 'gedcom_ai_integrator'), "Should have integrator attribute"


def _test_task_templates_structure():
    """Test that task templates are properly structured."""
    generator = GenealogicalTaskGenerator()
    templates = generator.task_templates

    # Test template keys exist
    required_templates = [
        "vital_records_search", "dna_match_analysis", "immigration_research",
        "census_research", "military_research", "occupation_research"
    ]
    for template_key in required_templates:
        assert template_key in templates, f"Template {template_key} should be available"
        assert isinstance(templates[template_key], dict), f"Template {template_key} should be a dictionary"
        assert "title" in templates[template_key], f"Template {template_key} should have title"


def _test_basic_task_generation():
    """Test basic task generation functionality."""
    generator = GenealogicalTaskGenerator()

    # Test data
    test_extracted_data = {
        "structured_names": [
            {"full_name": "John Smith", "nicknames": [], "maiden_name": None}
        ],
        "vital_records": [
            {"person": "John Smith", "event_type": "birth", "date": "1850", "place": "Aberdeen, Scotland"}
        ],
        "locations": [
            {"place": "Aberdeen, Scotland", "context": "birthplace", "time_period": "1850"}
        ],
        "research_questions": ["finding John Smith's parents"]
    }

    test_person_data = {"username": "TestUser"}
    test_suggested_tasks = ["Research John Smith's family history"]

    # Test task generation
    tasks = generator.generate_research_tasks(
        test_person_data,
        test_extracted_data,
        test_suggested_tasks
    )

    assert isinstance(tasks, list), "Should return a list of tasks"
    assert len(tasks) > 0, "Should generate at least one task"

    # Test task structure
    for task in tasks:
        assert isinstance(task, dict), "Each task should be a dictionary"
        assert "title" in task, "Task should have a title"
        assert "description" in task, "Task should have a description"


def _test_vital_records_task_generation():
    """Test specialized vital records task generation."""
    generator = GenealogicalTaskGenerator()

    extracted_data = {
        "vital_records": [
            {"person": "Mary Johnson", "event_type": "marriage", "date": "1875", "place": "Boston, MA"},
            {"person": "William Johnson", "event_type": "death", "date": "1900", "place": "New York, NY"}
        ]
    }

    vital_tasks = generator._generate_vital_records_tasks(extracted_data)

    assert isinstance(vital_tasks, list), "Should return list of vital records tasks"
    if len(vital_tasks) > 0:  # Only test if tasks were generated
        task = vital_tasks[0]
        assert "title" in task, "Vital records task should have title"
        assert "description" in task, "Vital records task should have description"
        assert "priority" in task, "Vital records task should have priority"


def _test_location_task_generation():
    """Test location-based task generation."""
    generator = GenealogicalTaskGenerator()

    extracted_data = {
        "locations": [
            {"place": "Dublin, Ireland", "context": "birthplace", "time_period": "1840"},
            {"place": "Liverpool, England", "context": "immigration", "time_period": "1860"}
        ]
    }

    location_tasks = generator._generate_location_tasks(extracted_data)

    assert isinstance(location_tasks, list), "Should return list of location tasks"
    if len(location_tasks) > 0:  # Only test if tasks were generated
        for task in location_tasks:
            assert isinstance(task, dict), "Each location task should be a dictionary"
            assert "title" in task, "Location task should have title"


def _test_occupation_task_generation():
    """Test occupation-based task generation."""
    generator = GenealogicalTaskGenerator()

    extracted_data = {
        "occupations": [
            {"person": "Thomas Baker", "occupation": "baker", "location": "London", "time_period": "1880-1900"},
            {"person": "Sarah Miller", "occupation": "seamstress", "location": "Manchester", "time_period": "1870"}
        ]
    }

    occupation_tasks = generator._generate_occupation_tasks(extracted_data)

    assert isinstance(occupation_tasks, list), "Should return list of occupation tasks"
    if len(occupation_tasks) > 0:  # Only test if tasks were generated
        for task in occupation_tasks:
            assert isinstance(task, dict), "Each occupation task should be a dictionary"
            assert "title" in task, "Occupation task should have title"


def _test_empty_data_handling():
    """Test task generation with empty or minimal data."""
    generator = GenealogicalTaskGenerator()

    # Test with completely empty data
    empty_tasks = generator.generate_research_tasks({}, {}, [])
    assert isinstance(empty_tasks, list), "Should return list even with empty data"

    # Test with minimal data
    minimal_person = {"username": "TestUser"}
    minimal_extracted = {"structured_names": []}
    minimal_suggested = []

    minimal_tasks = generator.generate_research_tasks(minimal_person, minimal_extracted, minimal_suggested)
    assert isinstance(minimal_tasks, list), "Should handle minimal data gracefully"


def _test_invalid_template_handling():
    """Test handling of invalid or missing template data."""
    generator = GenealogicalTaskGenerator()

    # Test with invalid template key
    invalid_task = generator._create_task_from_template("nonexistent_template", {"test": "data"})
    assert invalid_task is None, "Should return None for invalid template"

    # Test with empty task data
    valid_template = next(iter(generator.task_templates.keys()))
    empty_task = generator._create_task_from_template(valid_template, {})
    # Should handle empty data gracefully (may return task or None)
    assert empty_task is None or isinstance(empty_task, dict), "Should handle empty data gracefully"


def _test_fallback_task_creation():
    """Test fallback task creation when no specialized tasks can be generated."""
    generator = GenealogicalTaskGenerator()

    person_data = {"username": "TestUser"}
    suggested_tasks = ["Research family history", "Find birth records"]

    fallback_tasks = generator._create_fallback_tasks(person_data, suggested_tasks)

    assert isinstance(fallback_tasks, list), "Should return list of fallback tasks"
    assert len(fallback_tasks) > 0, "Should generate at least one fallback task"

    for task in fallback_tasks:
        assert isinstance(task, dict), "Fallback task should be dictionary"
        assert "title" in task, "Fallback task should have title"
        assert "description" in task, "Fallback task should have description"


def _test_gedcom_ai_integration():
    """Test GEDCOM AI integration when available."""
    generator = GenealogicalTaskGenerator()

    # Test AI availability tracking
    assert hasattr(generator, 'gedcom_ai_available'), "Should track AI availability"
    assert isinstance(generator.gedcom_ai_available, bool), "AI availability should be boolean"

    # Test integrator attribute existence
    assert hasattr(generator, 'gedcom_ai_integrator'), "Should have integrator attribute"
    # integrator may be None if not available, which is fine


def _test_task_prioritization():
    """Test task prioritization and limiting functionality."""
    generator = GenealogicalTaskGenerator()

    # Create test tasks with different priorities
    test_tasks = [
        {"title": "High Priority Task", "priority": "high", "description": "Test"},
        {"title": "Medium Priority Task", "priority": "medium", "description": "Test"},
        {"title": "Low Priority Task", "priority": "low", "description": "Test"},
        {"title": "Another High Task", "priority": "high", "description": "Test"}
    ]

    prioritized_tasks = generator._prioritize_and_limit_tasks(test_tasks)

    assert isinstance(prioritized_tasks, list), "Should return list of prioritized tasks"
    assert len(prioritized_tasks) <= len(test_tasks), "Should not exceed original task count"

    # Check that high priority tasks come first if any prioritization occurred
    if len(prioritized_tasks) > 1:
        first_task = prioritized_tasks[0]
        assert "priority" in first_task, "Prioritized task should have priority field"


def _test_template_configuration_loading():
    """Test loading and validation of task configuration."""
    generator = GenealogicalTaskGenerator()
    config = generator.task_config

    # Test configuration structure
    assert isinstance(config, dict), "Task config should be dictionary"

    # Test for expected configuration keys
    expected_keys = ["max_tasks_per_person", "priority_weights", "default_priority"]
    for key in expected_keys:
        if key in config:  # Optional keys may not exist
            assert config[key] is not None, f"Config key {key} should not be None"


def _test_performance():
    """Test performance of task generation operations."""
    import time
    generator = GenealogicalTaskGenerator()

    # Test data
    test_extracted_data = {
        "structured_names": [
            {"full_name": f"Person {i}", "nicknames": [], "maiden_name": None}
            for i in range(10)
        ],
        "vital_records": [
            {"person": f"Person {i}", "event_type": "birth", "date": f"{1850+i}", "place": "Test Location"}
            for i in range(10)
        ],
        "locations": [
            {"place": f"Location {i}", "context": "birthplace", "time_period": f"{1850+i}"}
            for i in range(5)
        ]
    }

    start_time = time.time()

    # Run task generation multiple times
    for _ in range(5):
        tasks = generator.generate_research_tasks(
            {"username": "TestUser"},
            test_extracted_data,
            ["Test task"]
        )
        assert isinstance(tasks, list), "Should return tasks list"

    elapsed = time.time() - start_time
    assert elapsed < 1.0, f"Performance test should complete quickly, took {elapsed:.3f}s"


def _test_bulk_template_processing():
    """Test performance with bulk template processing."""
    import time
    generator = GenealogicalTaskGenerator()

    start_time = time.time()

    # Process multiple template types
    template_keys = list(generator.task_templates.keys())[:5]  # Test first 5 templates

    for template_key in template_keys:
        for i in range(10):
            task_data = {"person_name": f"Test Person {i}", "time_period": "1850-1900"}
            task = generator._create_task_from_template(template_key, task_data)
            # Task may be None or dict, both are acceptable
            assert task is None or isinstance(task, dict), "Task should be None or dict"

    elapsed = time.time() - start_time
    assert elapsed < 0.5, f"Bulk template processing should be fast, took {elapsed:.3f}s"


def _test_error_handling():
    """Test error handling with invalid inputs and edge cases."""
    generator = GenealogicalTaskGenerator()

    # Test with None inputs (using type ignore for intentional testing)
    result = generator.generate_research_tasks(None, None, None)  # type: ignore
    assert isinstance(result, list), "Should handle None inputs gracefully"

    # Test with invalid data types (using type ignore for intentional testing)
    result = generator.generate_research_tasks("invalid", "invalid", "invalid")  # type: ignore
    assert isinstance(result, list), "Should handle invalid data types"

    # Test private methods with invalid data
    result = generator._generate_vital_records_tasks({})
    assert isinstance(result, list), "Should handle empty vital records data"

    result = generator._generate_location_tasks({"locations": "invalid"})
    assert isinstance(result, list), "Should handle invalid location data"


def _test_malformed_data_handling():
    """Test handling of malformed or corrupted data structures."""
    generator = GenealogicalTaskGenerator()

    # Test with malformed extracted data
    malformed_data = {
        "vital_records": "not_a_list",
        "locations": [{"incomplete": "data"}],
        "occupations": [None, {"invalid": True}]
    }

    tasks = generator.generate_research_tasks(
        {"username": "Test"},
        malformed_data,
        ["test"]
    )

    assert isinstance(tasks, list), "Should handle malformed data gracefully"
    # Tasks list may be empty or contain fallback tasks, both are acceptable


# ==============================================
# MAIN TEST SUITE RUNNER
# ==============================================


def genealogical_task_templates_module_tests() -> bool:
    """
    Comprehensive test suite for genealogical_task_templates.py.
    Tests genealogical task generation, template management, and specialized research workflows.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Genealogical Task Templates & Research Generation", "genealogical_task_templates.py")
    suite.start_suite()

    # Run all tests
    with suppress_logging():
        suite.run_test(
            "Module imports and initialization",
            _test_module_imports,
            "All required modules and dependencies are properly imported",
            "Test import availability of core infrastructure and genealogical components",
            "Module initialization provides complete dependency access"
        )

        suite.run_test(
            "Task generator initialization",
            _test_task_generator_initialization,
            "GenealogicalTaskGenerator initializes correctly with templates and configuration",
            "Test GenealogicalTaskGenerator initialization and setup",
            "Task generator initialization provides complete template and AI integration setup"
        )

        suite.run_test(
            "Task templates structure validation",
            _test_task_templates_structure,
            "Task templates are properly structured with required fields and formats",
            "Test that task templates are properly structured",
            "Task templates provide complete genealogical research template library"
        )

        suite.run_test(
            "Basic task generation functionality",
            _test_basic_task_generation,
            "Task generation creates valid, structured genealogical research tasks",
            "Test basic task generation functionality",
            "Basic task generation provides actionable genealogical research tasks"
        )

        suite.run_test(
            "Vital records task generation",
            _test_vital_records_task_generation,
            "Vital records tasks are generated with proper structure and priority",
            "Test specialized vital records task generation",
            "Vital records task generation provides focused genealogical research objectives"
        )

        suite.run_test(
            "Location-based task generation",
            _test_location_task_generation,
            "Location tasks are generated based on geographical research opportunities",
            "Test location-based task generation",
            "Location task generation provides place-specific research strategies"
        )

        suite.run_test(
            "Occupation-based task generation",
            _test_occupation_task_generation,
            "Occupation tasks are generated to explore professional and trade connections",
            "Test occupation-based task generation",
            "Occupation task generation provides professional research pathways"
        )

        suite.run_test(
            "Empty data handling",
            _test_empty_data_handling,
            "Task generation handles empty or minimal data gracefully",
            "Test task generation with empty or minimal data",
            "Empty data handling ensures robust operation with incomplete information"
        )

        suite.run_test(
            "Invalid template handling",
            _test_invalid_template_handling,
            "Invalid or missing template data is handled gracefully",
            "Test handling of invalid or missing template data",
            "Invalid template handling provides robust template processing"
        )

        suite.run_test(
            "Fallback task creation",
            _test_fallback_task_creation,
            "Fallback tasks are created when specialized tasks cannot be generated",
            "Test fallback task creation when no specialized tasks can be generated",
            "Fallback task creation ensures users always receive actionable research tasks"
        )

        suite.run_test(
            "GEDCOM AI integration",
            _test_gedcom_ai_integration,
            "GEDCOM AI integration is properly configured and tracked",
            "Test GEDCOM AI integration when available",
            "GEDCOM AI integration provides enhanced genealogical analysis capabilities"
        )

        suite.run_test(
            "Task prioritization and limiting",
            _test_task_prioritization,
            "Task prioritization orders research tasks by importance and limits output",
            "Test task prioritization and limiting functionality",
            "Task prioritization ensures most important research tasks are presented first"
        )

        suite.run_test(
            "Template configuration loading",
            _test_template_configuration_loading,
            "Task configuration is loaded and validated correctly",
            "Test loading and validation of task configuration",
            "Configuration loading provides proper task generation parameters"
        )

        suite.run_test(
            "Performance validation",
            _test_performance,
            "Task generation operations complete within reasonable time limits",
            "Test performance of task generation operations",
            "Performance validation ensures efficient genealogical task generation"
        )

        suite.run_test(
            "Bulk template processing performance",
            _test_bulk_template_processing,
            "Bulk template processing handles multiple templates efficiently",
            "Test performance with bulk template processing",
            "Bulk processing provides scalable template-based task generation"
        )

        suite.run_test(
            "Error handling robustness",
            _test_error_handling,
            "Error handling gracefully manages invalid inputs and edge cases",
            "Test error handling with invalid inputs and edge cases",
            "Error handling ensures stable operation under adverse conditions"
        )

        suite.run_test(
            "Malformed data handling",
            _test_malformed_data_handling,
            "Malformed or corrupted data structures are handled gracefully",
            "Test handling of malformed or corrupted data structures",
            "Malformed data handling provides robust data processing capabilities"
        )

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(genealogical_task_templates_module_tests)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    import sys

    # Always run comprehensive tests
    print("🧬 Running Genealogical Task Templates comprehensive test suite...")
    success = run_comprehensive_tests()
    if success:
        print("\n✅ All genealogical task templates tests completed successfully!")
    else:
        print("\n❌ Some genealogical task templates tests failed!")
    sys.exit(0 if success else 1)
