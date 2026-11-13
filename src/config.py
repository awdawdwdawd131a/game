"""
Configuration, constants, weather presets, and round modifiers
for Crimean Bridge 1v1 (FPV Drone vs Fuel Truck).
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Network & Simulation Constants
# ---------------------------------------------------------------------------
DEFAULT_PORT = 27015
TICK_RATE = 60
DT = 1.0 / TICK_RATE
MAX_ROUNDS_TO_WIN = 3        # Best of 5 (first to 3 wins)
ROUND_TIME_LIMIT = 95.0      # Seconds per round before truck auto-survives if close or timeout
MODIFIER_PICK_TIME = 30.0    # Seconds in pre-round random loadout briefing phase
ROUND_END_DELAY = 4.5        # Seconds showing round winner banner

# ---------------------------------------------------------------------------
# Bridge Geometry Constants (3D coordinates: X=across width, Y=up, Z=along bridge)
# ---------------------------------------------------------------------------
BRIDGE_LENGTH = 1500.0       # Z from -750 to +750 meters
BRIDGE_START_Z = -560.0
BRIDGE_FINISH_Z = 680.0
ROAD_DECK_Y = 18.0           # Roadway elevation above sea level (Y=0 is sea)
ROAD_HALF_WIDTH = 13.0       # 4-lane highway width = 26m (-13..+13)
RAIL_DECK_Y = 22.0           # Parallel railway bridge deck height
RAIL_CENTER_X = 28.0         # Railway bridge offset to the side (+X)
RAIL_HALF_WIDTH = 6.5

ARCH_CENTER_Z = 0.0          # Iconic white shipping arches in the middle of the span
ARCH_HALF_SPAN_Z = 115.0     # Arches span from Z=-115 to Z=+115 (230m main span)
ARCH_PEAK_Y = 63.0           # 45m above deck (18 + 45 = 63m)
PILLAR_SPACING = 110.0

# Cover barriers / checkpoints placed periodically on the road
NUM_BARRIERS = 14

# ---------------------------------------------------------------------------
# Base Vehicle & Drone Stats (Rebalanced for fair 1v1 defense & real ballistics)
# ---------------------------------------------------------------------------
TRUCK_BASE_HP = 125.0
TRUCK_BASE_MAX_SPEED = 23.5      # m/s (~85 km/h)
TRUCK_BASE_ACCEL = 13.0          # m/s^2
TRUCK_BASE_BRAKE = 24.0          # m/s^2
TRUCK_STEER_RATE = 1.75          # rad/s max steer
TRUCK_LENGTH = 11.5
TRUCK_WIDTH = 3.1
TRUCK_HEIGHT = 3.7

# Truck Anti-FPV Weapons (Physical 3D projectile ballistics!)
SHOTGUN_COOLDOWN = 0.56          # Semi-auto anti-drone shotgun
SHOTGUN_PELLETS = 14
SHOTGUN_SPREAD_DEG = 4.5
SHOTGUN_RANGE = 115.0            # Effective close/mid-range buckshot envelope
SHOTGUN_PELLET_DMG = 9.0
SHOTGUN_PELLET_SPEED = 250.0     # m/s muzzle velocity (subject to high air drag & gravity)
SHOTGUN_PELLET_DRAG = 0.78

TURRET_FIRE_INTERVAL = 0.12      # Heavy roof machinegun / rapid AA (~500 rpm)
TURRET_HEAT_PER_SHOT = 0.045
TURRET_COOL_RATE = 0.38
TURRET_RANGE = 260.0             # Ballistic range; bullets drop & slow down with distance
TURRET_BULLET_SPEED = 360.0      # m/s muzzle velocity (requires real lead against moving FPV)
TURRET_BULLET_DRAG = 0.16
TURRET_DMG = 17.0                # About six hits down a stock 100 HP FPV
TURRET_LOCK_TIME = 0.65          # Ballistic FCS tracking buildup (no damage nerf on hits!)
TURRET_LOCK_CONE = 0.22          # radians, half-angle of tracking cone

EW_JAMMER_DURATION = 5.0         # Electronic Warfare (РЭБ) active pulse
EW_JAMMER_COOLDOWN = 11.0
EW_JAMMER_RADIUS = 55.0

SMOKE_DURATION = 6.5             # Aerosol smoke screen
SMOKE_COOLDOWN = 12.0
SMOKE_RADIUS = 34.0

# FPV Drone Base Stats (Challenging realistic quadcopter flight dynamics)
DRONE_BASE_HP = 100.0
DRONE_BASE_LIVES = 3             # 3 FPV drones in operator's quiver per round
DRONE_RESPAWN_DELAY = 3.2
DRONE_MAX_SPEED = 54.0           # m/s (~194 km/h fast FPV)
DRONE_THRUST = 38.0              # m/s^2 assisted flight thrust
DRONE_GRAVITY = 9.81
DRONE_DRAG = 0.42                # Classic assisted FPV response
DRONE_BATTERY_SECONDS = 45.0     # Flight time per drone before battery dies
DRONE_EXPLOSION_RADIUS = 13.5
DRONE_DIRECT_DAMAGE = 66.0       # 2 clean direct kamikaze strikes destroy a stock 125 HP truck!
DRONE_DETONATE_COOLDOWN = 4.0    # Re-arm time of the warhead after a stand-off airburst
DRONE_TRIGGER_DIST = 4.2         # Proximity / impact detonation radius


# ---------------------------------------------------------------------------
# Weather System (Different weather every round!)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeatherPreset:
    id: str
    name_ru: str
    description_ru: str
    sky_top: Tuple[int, int, int]
    sky_horizon: Tuple[int, int, int]
    sea_color: Tuple[int, int, int]
    fog_color: Tuple[int, int, int]
    fog_start: float
    fog_end: float
    ambient_light: float
    wind_vec: Tuple[float, float, float]   # Base wind pushing FPV drone (m/s)
    gust_strength: float                   # Dynamic turbulence amplitude
    truck_grip: float                      # Road tire grip multiplier (1.0 normal, <1 slippery)
    rain_intensity: float                  # 0..1 rain streaks
    lightning_chance: float                # Per-second flash chance
    drone_battery_drain_mult: float        # Cold/wind drain factor
    # --- Небо и окружение (рендер) -------------------------------------------
    star_density: float = 0.0              # 0..1 - плотность звёзд (только ночью)
    cloud_amount: float = 0.0              # 0..1 - количество облаков
    cloud_color: Tuple[int, int, int] = (200, 205, 215)
    sun_glow: float = 0.0                  # 0..1 - видимость диска светила
    sun_color: Tuple[int, int, int] = (255, 238, 190)
    sun_azimuth: float = 0.0               # азимут светила в мире (рад, 0 = +Z)
    sun_elevation: float = 0.55            # высота светила над горизонтом (рад)
    city_lights: float = 0.0               # 0..1 - огни города на том берегу
    sea_wave: float = 0.35                 # 0..1 - амплитуда бликов на воде


WEATHER_PRESETS: List[WeatherPreset] = [
    WeatherPreset(
        id="clear_day",
        name_ru="Ясный крымский полдень",
        description_ru="Отличная видимость, лёгкий морской бриз, сухой асфальт.",
        sky_top=(42, 108, 198),
        sky_horizon=(158, 206, 245),
        sea_color=(22, 82, 136),
        fog_color=(165, 208, 242),
        fog_start=320.0,
        fog_end=980.0,
        ambient_light=1.0,
        wind_vec=(2.2, 0.0, -1.0),
        gust_strength=1.2,
        truck_grip=1.0,
        rain_intensity=0.0,
        lightning_chance=0.0,
        drone_battery_drain_mult=1.0,
        star_density=0.0,
        cloud_amount=0.45,
        cloud_color=(252, 252, 254),
        sun_glow=0.8,
        sun_color=(255, 250, 226),
        sun_azimuth=3.45,
        sun_elevation=0.55,
        city_lights=0.0,
        sea_wave=0.55,
    ),
    WeatherPreset(
        id="kerch_fog",
        name_ru="Густой керченский туман",
        description_ru="Видимость сильно ограничена! Дрону сложнее искать цель, но грузовик видит угрозу в последний момент.",
        sky_top=(118, 130, 140),
        sky_horizon=(176, 186, 194),
        sea_color=(68, 86, 98),
        fog_color=(178, 188, 195),
        fog_start=18.0,
        fog_end=125.0,
        ambient_light=0.78,
        wind_vec=(-1.5, 0.2, 0.8),
        gust_strength=1.0,
        truck_grip=0.90,
        rain_intensity=0.1,
        lightning_chance=0.0,
        drone_battery_drain_mult=1.05,
        star_density=0.0,
        cloud_amount=0.6,
        cloud_color=(206, 210, 214),
        sun_glow=0.18,
        sun_color=(226, 230, 234),
        sun_azimuth=3.15,
        sun_elevation=0.4,
        city_lights=0.0,
        sea_wave=0.12,
    ),
    WeatherPreset(
        id="black_sea_storm",
        name_ru="Черноморский шторм и гроза",
        description_ru="Ливень, вспышки молний, мокрый скользкий асфальт и сильная турбулентность для FPV.",
        sky_top=(24, 30, 46),
        sky_horizon=(58, 68, 86),
        sea_color=(18, 34, 48),
        fog_color=(56, 66, 82),
        fog_start=95.0,
        fog_end=430.0,
        ambient_light=0.52,
        wind_vec=(8.5, -1.2, 4.0),
        gust_strength=6.5,
        truck_grip=0.68,
        rain_intensity=1.0,
        lightning_chance=0.22,
        drone_battery_drain_mult=1.18,
        star_density=0.0,
        cloud_amount=0.92,
        cloud_color=(58, 64, 74),
        sun_glow=0.05,
        sun_color=(120, 126, 136),
        sun_azimuth=2.9,
        sun_elevation=0.28,
        city_lights=0.35,
        sea_wave=0.85,
    ),
    WeatherPreset(
        id="crimson_sunset",
        name_ru="Багровый закат и боковой шквал",
        description_ru="Слепящий закат над проливом и резкий боковой ветер с Азовского моря, сносящий дрон к аркам.",
        sky_top=(64, 36, 88),
        sky_horizon=(238, 112, 58),
        sea_color=(58, 42, 74),
        fog_color=(215, 118, 74),
        fog_start=180.0,
        fog_end=720.0,
        ambient_light=0.82,
        wind_vec=(12.5, 0.6, -2.0),
        gust_strength=5.2,
        truck_grip=0.95,
        rain_intensity=0.0,
        lightning_chance=0.0,
        drone_battery_drain_mult=1.10,
        star_density=0.15,
        cloud_amount=0.72,
        cloud_color=(226, 138, 108),
        sun_glow=1.0,
        sun_color=(255, 196, 104),
        sun_azimuth=3.6,
        sun_elevation=0.14,
        city_lights=0.0,
        sea_wave=0.7,
    ),
    WeatherPreset(
        id="night_spotlight",
        name_ru="Ночная тревога (Прожекторы)",
        description_ru="Темнота над проливом. Видны фонари моста и фары бензовоза.",
        sky_top=(8, 12, 24),
        sky_horizon=(20, 28, 48),
        sea_color=(6, 12, 22),
        fog_color=(16, 22, 38),
        fog_start=80.0,
        fog_end=390.0,
        ambient_light=0.34,
        wind_vec=(-3.5, 0.0, -2.5),
        gust_strength=2.4,
        truck_grip=0.92,
        rain_intensity=0.0,
        lightning_chance=0.0,
        drone_battery_drain_mult=1.12,
        star_density=0.95,
        cloud_amount=0.3,
        cloud_color=(30, 36, 52),
        sun_glow=0.9,
        sun_color=(226, 232, 246),
        sun_azimuth=3.3,
        sun_elevation=0.62,
        city_lights=0.85,
        sea_wave=0.3,
    ),
    WeatherPreset(
        id="winter_gale",
        name_ru="Ледяной норд-ост (Бора)",
        description_ru="Шквальный ледяной ветер, снежная крупа, гололёд на мосту и ускоренный разряд батареи FPV.",
        sky_top=(88, 108, 132),
        sky_horizon=(190, 205, 220),
        sea_color=(36, 58, 78),
        fog_color=(182, 198, 214),
        fog_start=110.0,
        fog_end=480.0,
        ambient_light=0.86,
        wind_vec=(-11.0, -0.8, 5.5),
        gust_strength=7.0,
        truck_grip=0.58,
        rain_intensity=0.65,
        lightning_chance=0.0,
        drone_battery_drain_mult=1.35,
        star_density=0.0,
        cloud_amount=0.85,
        cloud_color=(216, 222, 230),
        sun_glow=0.22,
        sun_color=(228, 234, 242),
        sun_azimuth=3.0,
        sun_elevation=0.45,
        city_lights=0.0,
        sea_wave=0.6,
    ),
]

# Additional conditions share the same physical and visual weather model.
WEATHER_PRESETS.extend([
    replace(WEATHER_PRESETS[1], id="sea_mist", name_ru="Морской туман на рассвете",
            description_ru="Очень плотная морось и туман скрывают мост уже вблизи.",
            sky_top=(100, 119, 128), sky_horizon=(157, 175, 181), fog_color=(158, 176, 181),
            fog_start=8.0, fog_end=85.0, wind_vec=(1.5, 0.0, 0.5), rain_intensity=0.25),
    replace(WEATHER_PRESETS[2], id="cold_rain", name_ru="Холодный осенний дождь",
            description_ru="Серое небо, дождь и скользкая дорога.", sky_top=(72, 82, 92),
            sky_horizon=(123, 135, 145), fog_color=(128, 139, 148), fog_start=55.0,
            fog_end=240.0, wind_vec=(4.0, 0.0, 2.0), gust_strength=3.0,
            rain_intensity=0.7, lightning_chance=0.0, ambient_light=0.68),
    replace(WEATHER_PRESETS[4], id="foggy_night", name_ru="Туманная ночь",
            description_ru="Ночь и густой туман: цель видна только вблизи.",
            fog_start=15.0, fog_end=115.0, star_density=0.0, cloud_amount=0.85,
            sun_glow=0.1, city_lights=0.15, wind_vec=(-1.0, 0.0, 0.5)),
])

WEATHER_BY_ID: Dict[str, WeatherPreset] = {w.id: w for w in WEATHER_PRESETS}


# ---------------------------------------------------------------------------
# Expanded Modifiers Pool (Issued Randomly Each Round!)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModifierDef:
    id: str
    role: str              # "drone" or "truck"
    name_ru: str
    short_ru: str
    desc_ru: str
    key_hint: str = ""


DRONE_MODIFIERS: List[ModifierDef] = [
    ModifierDef(
        id="shaped_charge",
        role="drone",
        name_ru="Кумулятивная БЧ (ПГ-7ВЛ)",
        short_ru="+40% урона БЧ, -12% скорости и тяги",
        desc_ru="Тяжёлая граната РПГ под рамой: мощнейший удар по броне цистерны, но дрон тяжелее и инертнее.",
    ),
    ModifierDef(
        id="fiber_optic",
        role="drone",
        name_ru="Оптоволокно (Иммунитет к РЭБ)",
        short_ru="Полная защита от РЭБ, +20% батареи, -6% скорости",
        desc_ru="Катушка оптоволокна: РЭБ бензовоза вообще не глушит видео и рули, но катушка даёт лёгкое сопротивление.",
    ),
    ModifierDef(
        id="sport_motors",
        role="drone",
        name_ru="Форсажные бесколлекторники 6S",
        short_ru="+25% тяги и макс. скорости, мощный форсаж",
        desc_ru="Гоночные моторы: молниеносное пикирование и уклонение от трассеров, но высокий расход батареи на форсаже.",
    ),
    ModifierDef(
        id="kevlar_frame",
        role="drone",
        name_ru="Кевларовая рама + Доп. борт",
        short_ru="+1 запасной FPV дрон (4 всего) и +45% HP рамы",
        desc_ru="Усиленная карбон-кевларовая рама держит больше попаданий пуль и дроби, плюс дополнительный борт в запасе.",
    ),
    ModifierDef(
        id="airburst_frag",
        role="drone",
        name_ru="Осколочный подрыв",
        short_ru="+55% радиус взрыва",
        desc_ru="Мощный осколочный разлёт накрывает бензовоз с дистанции.",
    ),
    ModifierDef(
        id="stealth_matte",
        role="drone",
        name_ru="Стелс-рама (Малозаметный профиль)",
        short_ru="-30% хитбокс под обстрелом, скрыт от радара >45м",
        desc_ru="Матовый мини-корпус и тихие винты: труднее зацепить пулей и не подсвечивается меткой у стрелка на дистанции.",
    ),
    ModifierDef(
        id="wind_gyro",
        role="drone",
        name_ru="Цифровой гиростабилизатор ветра",
        short_ru="-75% сноса от ветра и шквала, +25% точность рулей",
        desc_ru="Продвинутый полётный контроллер компенсирует боковой ветер и турбулентность даже в бурю и бору.",
    ),
    ModifierDef(
        id="tandem_warhead",
        role="drone",
        name_ru="Тандемная БЧ (Пробив «Мангала»)",
        short_ru="Игнорирует антидрон-сетку, +20% урона в упор",
        desc_ru="Лидерный предзаряд прожигает стальной «мангал» и наносит полный урон основной струёй прямо в цистерну.",
    ),
    ModifierDef(
        id="cluster_munitions",
        role="drone",
        name_ru="Кассетный сброс (3 суббоеприпаса)",
        short_ru="Быстрый откат БЧ (2.2с), сброс 3 падающих гранат",
        desc_ru="При дистанционном подрыве сбрасывает 3 падающие гранаты с разлётом по площади — легко задеть маневрирующую цель.",
    ),
    ModifierDef(
        id="long_range_lipo",
        role="drone",
        name_ru="Батарея Li-Ion 8000mAh + Усиление",
        short_ru="+75% времени полёта и +25% HP рамы",
        desc_ru="Огромный запас энергии для затяжного выжидания ошибки стрелка и многократных заходов с высоты.",
    ),
]

TRUCK_MODIFIERS: List[ModifierDef] = [
    ModifierDef(
        id="anti_drone_cage",
        role="truck",
        name_ru="Антидроновый «Мангал» и сетка",
        short_ru="-35% урона от БЧ и +20 к макс. броне (145 HP)",
        desc_ru="Стальной каркас над кабиной и цистерной взводит заряд раньше времени, резко снижая фугасный урон.",
    ),
    ModifierDef(
        id="ew_kupol",
        role="truck",
        name_ru="Мощный РЭБ «Купол-М»",
        short_ru="+35% радиус РЭБ (74м), дольше работа и быстрее откат",
        desc_ru="Широкополосная глушилка засыпает экран оператора помехами и уводит рули радио-дрона на подлёте.",
    ),
    ModifierDef(
        id="twin_zu_turret",
        role="truck",
        name_ru="Спаренная зенитная турель ЗУ-23",
        short_ru="Почти вдвое выше темп огня, -40% нагрева, +15% скорость пуль",
        desc_ru="Два ствола создают плотную свинцовую стену в небе с высокой начальной скоростью пули.",
    ),
    ModifierDef(
        id="v12_biturbo",
        role="truck",
        name_ru="Форсированный V12 + Шипы",
        short_ru="+25% макс. скорости (~106 км/ч), +35% разгона и зацеп",
        desc_ru="Тягач рвёт к финишу на пределе и уверенно держит дорогу даже в шторм и гололёд.",
    ),
    ModifierDef(
        id="buckshot_radar",
        role="truck",
        name_ru="Картечь «Перехват» + Радар ПВО",
        short_ru="28 дробин, упреждение на 240м и быстрый дым",
        desc_ru="Двойной заряд вольфрамовой дроби срубает пикирующий FPV одним залпом, а радар рисует точку упреждения.",
    ),
    ModifierDef(
        id="ballistic_computer",
        role="truck",
        name_ru="Баллистический вычислитель ПВО",
        short_ru="Точное упреждение до 260м, +25% скорость пуль, -25% разброс",
        desc_ru="Оптико-лазерный прицел точно просчитывает падение и скорость пули на 450 м/с для снайперского перехвата.",
    ),
    ModifierDef(
        id="reactive_armor",
        role="truck",
        name_ru="Динамическая защита «Контакт»",
        short_ru="Первый удар БЧ гасится на 65% + +15 к броне (140 HP)",
        desc_ru="Блоки ДЗ на цистерне отбивают первую атаку FPV встречным импульсом, спасая машину в критический момент.",
    ),
    ModifierDef(
        id="flak_shrapnel",
        role="truck",
        name_ru="Зенитные снаряды с радиовзрывателем",
        short_ru="+55% радиус поражения пуль (воздушный подрыв)",
        desc_ru="Пули турели подрываются при пролёте рядом с рамой FPV, сбивая дрон даже при близких промахах.",
    ),
    ModifierDef(
        id="auto_extinguisher",
        role="truck",
        name_ru="Автопожаротушение + Усиленный бак",
        short_ru="+35 к броне (160 HP), авторемонт +2.5 HP/с",
        desc_ru="Пенная система и толстая стенка бака гасят последствия попаданий и постепенно латают цистерну на ходу.",
    ),
    ModifierDef(
        id="flamethrower_smoke",
        role="truck",
        name_ru="Термодымовая завеса ТДА",
        short_ru="Гигантская дымзавеса (2 облака, 9с) + быстрее откат РЭБ/дыма",
        desc_ru="Густой термодым накрывает участок моста за тягачом, скрывая бензовоз от обзора.",
    ),
]

DRONE_MOD_BY_ID: Dict[str, ModifierDef] = {m.id: m for m in DRONE_MODIFIERS}
TRUCK_MOD_BY_ID: Dict[str, ModifierDef] = {m.id: m for m in TRUCK_MODIFIERS}
