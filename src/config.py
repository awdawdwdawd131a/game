"""
Configuration, constants, weather presets, and round modifiers
for Crimean Bridge 1v1 (FPV Drone vs Fuel Truck).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Network & Simulation Constants
# ---------------------------------------------------------------------------
DEFAULT_PORT = 27015
TICK_RATE = 60
DT = 1.0 / TICK_RATE
MAX_ROUNDS_TO_WIN = 3        # Best of 5 (first to 3 wins)
ROUND_TIME_LIMIT = 95.0      # Seconds per round before truck auto-survives if close or timeout
MODIFIER_PICK_TIME = 18.0    # Seconds in pre-round draft phase
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
# Base Vehicle & Drone Stats
# ---------------------------------------------------------------------------
TRUCK_BASE_HP = 100.0
TRUCK_BASE_MAX_SPEED = 24.0      # m/s (~86 km/h)
TRUCK_BASE_ACCEL = 11.5          # m/s^2
TRUCK_BASE_BRAKE = 22.0          # m/s^2
TRUCK_STEER_RATE = 1.65          # rad/s max steer
TRUCK_LENGTH = 11.5
TRUCK_WIDTH = 3.1
TRUCK_HEIGHT = 3.7

# Truck Anti-FPV Weapons
SHOTGUN_COOLDOWN = 0.72          # Pump-action / semi-auto anti-drone shotgun
SHOTGUN_PELLETS = 11
SHOTGUN_SPREAD_DEG = 5.2
SHOTGUN_RANGE = 95.0             # Effective only at close range (anti-drone "Перехват")
SHOTGUN_PELLET_DMG = 6.0

TURRET_FIRE_INTERVAL = 0.14      # Heavy roof machinegun / rapid AA
TURRET_HEAT_PER_SHOT = 0.075
TURRET_COOL_RATE = 0.28
TURRET_RANGE = 170.0             # The drone must close in under the fire arc
TURRET_BULLET_SPEED = 260.0
TURRET_DMG = 7.0                 # ~14 hits to down a stock FPV frame
TURRET_LOCK_TIME = 0.80          # Time the gunner needs to keep the drone in the reticle
TURRET_LOCK_CONE = 0.18          # radians, half-angle of the tracking cone

EW_JAMMER_DURATION = 4.0         # Electronic Warfare (РЭБ) active pulse
EW_JAMMER_COOLDOWN = 14.0
EW_JAMMER_RADIUS = 85.0

SMOKE_DURATION = 5.5             # Aerosol smoke screen
SMOKE_COOLDOWN = 16.0
SMOKE_RADIUS = 28.0

# FPV Drone Base Stats
DRONE_BASE_HP = 100.0
DRONE_BASE_LIVES = 3             # 3 FPV drones in operator's quiver per round
DRONE_RESPAWN_DELAY = 2.5
DRONE_MAX_SPEED = 54.0           # m/s (~194 km/h fast FPV)
DRONE_THRUST = 38.0              # m/s^2 upward/forward thrust
DRONE_GRAVITY = 9.81
DRONE_DRAG = 0.42
DRONE_BATTERY_SECONDS = 55.0     # Flight time per drone before battery dies
DRONE_EXPLOSION_RADIUS = 13.5
DRONE_DIRECT_DAMAGE = 62.0       # 2 clean hits destroy base truck (or 1 shaped-charge hit + splash)
DRONE_DETONATE_COOLDOWN = 2.5    # Re-arm time of the warhead after a stand-off airburst
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
        fog_start=45.0,
        fog_end=250.0,
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
        city_lights=0.3,
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
        city_lights=0.4,
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
        city_lights=0.25,
        sea_wave=0.7,
    ),
    WeatherPreset(
        id="night_spotlight",
        name_ru="Ночная тревога (Прожекторы)",
        description_ru="Темнота над проливом. Освещение только от фонарей моста и фар бензовоза. Тепловизор дрона помогает!",
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
        city_lights=0.35,
        sea_wave=0.6,
    ),
]

WEATHER_BY_ID: Dict[str, WeatherPreset] = {w.id: w for w in WEATHER_PRESETS}


# ---------------------------------------------------------------------------
# Modifiers / Perks System for BOTH Players
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModifierDef:
    id: str
    role: str              # "drone" or "truck"
    name_ru: str
    short_ru: str
    desc_ru: str
    key_hint: str


DRONE_MODIFIERS: List[ModifierDef] = [
    ModifierDef(
        id="shaped_charge",
        role="drone",
        name_ru="Кумулятивная БЧ (ПГ-7ВЛ)",
        short_ru="+45% урона, -12% скорости",
        desc_ru="Тяжёлый выстрел РПГ под брюхом: огромный урон по цистерне, но выше масса и инерция.",
        key_hint="1",
    ),
    ModifierDef(
        id="fiber_optic",
        role="drone",
        name_ru="Оптоволокно (Иммунитет к РЭБ)",
        short_ru="Полная защита от РЭБ, +20% батареи",
        desc_ru="Управление по катушке оптоволокна: глушилка бензовоза не даёт помех и не блокирует рули.",
        key_hint="2",
    ),
    ModifierDef(
        id="sport_motors",
        role="drone",
        name_ru="Форсажные бесколлекторники",
        short_ru="+28% тяги и макс. скорости",
        desc_ru="Гоночные моторы и 6S LiPo: молниеносный пикирующий заход и кнопка нитро-рывка.",
        key_hint="3",
    ),
    ModifierDef(
        id="kevlar_frame",
        role="drone",
        name_ru="Кевларовая рама + Доп. борт",
        short_ru="+1 запасной FPV дрон и +45% HP",
        desc_ru="Усиленные лучи выдерживают часть дроби, плюс +1 дополнительный дрон в раунде.",
        key_hint="4",
    ),
    ModifierDef(
        id="airburst_frag",
        role="drone",
        name_ru="Осколочный подрыв + Тепловизор",
        short_ru="+55% радиус взрыва и подсветка цели",
        desc_ru="Дистанционный подрыв шрапнелью накрывает даже в дыму и тумане, контур цели виден сквозь мглу.",
        key_hint="5",
    ),
]

TRUCK_MODIFIERS: List[ModifierDef] = [
    ModifierDef(
        id="anti_drone_cage",
        role="truck",
        name_ru="Антидроновый «Мангал» и сетка",
        short_ru="-35% входящего урона от взрывов",
        desc_ru="Стальная решётка над кабиной и цистерной взводит гранату раньше времени, снижая урон.",
        key_hint="1",
    ),
    ModifierDef(
        id="ew_kupol",
        role="truck",
        name_ru="Мощный РЭБ «Купол-М»",
        short_ru="+60% радиус РЭБ, быстрее откат",
        desc_ru="Широкополосная станция помех глушит видеолинк FPV и уводит рули дрона на большой дистанции.",
        key_hint="2",
    ),
    ModifierDef(
        id="twin_zu_turret",
        role="truck",
        name_ru="Спаренная зенитная турель",
        short_ru="Двойной темп огня и -40% нагрева",
        desc_ru="Усиленная турель на крыше кабины с высокой плотностью огня и увеличенным уроном по FPV.",
        key_hint="3",
    ),
    ModifierDef(
        id="v12_biturbo",
        role="truck",
        name_ru="Форсированный V12 + Шипы",
        short_ru="+26% скорости, +35% разгона и зацепа",
        desc_ru="Тягач летит по мосту на максималке и уверенно держит трассу даже в гололёд и шторм.",
        key_hint="4",
    ),
    ModifierDef(
        id="buckshot_radar",
        role="truck",
        name_ru="Картечь «Перехват» + Радар + Дым",
        short_ru="Двойная картечь, радар дистанции и быстрый дым",
        desc_ru="Ружьё стреляет плотным облаком вольфрамовой дроби, а дымовые гранаты перезаряжаются вдвое быстрее.",
        key_hint="5",
    ),
]

DRONE_MOD_BY_ID: Dict[str, ModifierDef] = {m.id: m for m in DRONE_MODIFIERS}
TRUCK_MOD_BY_ID: Dict[str, ModifierDef] = {m.id: m for m in TRUCK_MODIFIERS}
