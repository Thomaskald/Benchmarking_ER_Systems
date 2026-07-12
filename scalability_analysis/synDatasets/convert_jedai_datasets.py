"""
convert_jedai_datasets.py
--------------------------
Converts JedAI Java-serialized synthetic datasets to CSV/Parquet
so they can be used by PyJedAI, Splink, Zingg, and Dedupe.

Files expected (from Zenodo 8433873):
  <size>profiles      → java.util.ArrayList of EntityProfile objects
  <size>IdDuplicates  → java.util.HashSet of duplicate ID pairs (ground truth)

Output per dataset size:
  <output_dir>/<size>/profiles.csv
  <output_dir>/<size>/ground_truth.csv

REQUIREMENTS:
  pip install pyjnius pandas pyarrow

  You also need:
    - Java (JDK 8+) installed:  sudo apt install default-jdk
    - JedAI jar downloaded:
        wget https://github.com/scify/JedAIToolkit/releases/download/v3.2/jedai-core-3.2-jar-with-dependencies.jar

Usage:
  # Basic (reads from ./data, writes to ./converted, uses jar in current dir)
  python3 convert_jedai_datasets.py

  # Custom paths
  python3 convert_jedai_datasets.py \
      --data_dir /home/thomas/train_test_valid_datasets/synDatasets/data \
      --output_dir ./converted \
      --jar ./jedai-core-3.2-jar-with-dependencies.jar \
      --format csv          # or parquet

  # Convert only specific sizes
  python3 convert_jedai_datasets.py --sizes 10K 50K 100K
"""

import os
import sys
import argparse
import struct
import traceback
from pathlib import Path

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert JedAI serialized datasets to CSV/Parquet"
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory containing the raw binary files (default: ./data)"
    )
    parser.add_argument(
        "--output_dir",
        default="./converted",
        help="Where to write CSV/Parquet outputs (default: ./converted)"
    )
    parser.add_argument(
        "--jar",
        default="./jedai-core-3.2-jar-with-dependencies.jar",
        help="Path to jedai-core-*-jar-with-dependencies.jar"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "both"],
        default="csv",
        help="Output format (default: csv)"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["10K", "50K", "100K", "200K", "300K", "1M", "2M"],
        help="Dataset sizes to convert (default: all)"
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use pure-Python fallback parser if pyjnius is unavailable"
    )
    return parser.parse_args()


# ── Pure-Python Java deserializer (fallback) ──────────────────────────────────
# Implements enough of the Java Object Serialization Protocol (JEP-1) to
# read ArrayList<EntityProfile> and HashSet<IdDuplicates> without Java/JVM.
# Reference: https://docs.oracle.com/javase/8/docs/platform/serialization/spec/protocol.html

class JavaDeserializer:
    """
    Minimal Java Object Serialization Protocol parser.
    Handles: ArrayList, HashSet, String, Integer, Long, EntityProfile, Attribute.
    """

    STREAM_MAGIC  = 0xACED
    STREAM_VERSION = 5

    TC_NULL       = 0x70
    TC_REFERENCE  = 0x71
    TC_CLASSDESC  = 0x72
    TC_OBJECT     = 0x73
    TC_STRING     = 0x74
    TC_ARRAY      = 0x75
    TC_CLASS      = 0x76
    TC_BLOCKDATA  = 0x77
    TC_ENDBLOCKDATA = 0x78
    TC_RESET      = 0x79
    TC_BLOCKDATALONG = 0x7A
    TC_EXCEPTION  = 0x7B
    TC_LONGSTRING = 0x7C
    TC_PROXYCLASSDESC = 0x7D
    TC_ENUM       = 0x7E

    SC_WRITE_METHOD = 0x01
    SC_BLOCK_DATA   = 0x08
    SC_SERIALIZABLE = 0x02
    SC_EXTERNALIZABLE = 0x04
    SC_ENUM         = 0x10

    def __init__(self, data):
        self.data   = data
        self.pos    = 0
        self.handles = []   # object reference table (base 0x7e0000)

    def read(self, n):
        chunk = self.data[self.pos:self.pos + n]
        if len(chunk) < n:
            raise EOFError(f"Expected {n} bytes at pos {self.pos}, got {len(chunk)}")
        self.pos += n
        return chunk

    def read_u1(self):
        return struct.unpack(">B", self.read(1))[0]

    def read_u2(self):
        return struct.unpack(">H", self.read(2))[0]

    def read_i4(self):
        return struct.unpack(">i", self.read(4))[0]

    def read_i8(self):
        return struct.unpack(">q", self.read(8))[0]

    def read_f4(self):
        return struct.unpack(">f", self.read(4))[0]

    def read_f8(self):
        return struct.unpack(">d", self.read(8))[0]

    def read_utf(self):
        length = self.read_u2()
        return self.read(length).decode("utf-8", errors="replace")

    def read_long_utf(self):
        length = struct.unpack(">Q", self.read(8))[0]
        return self.read(length).decode("utf-8", errors="replace")

    def new_handle(self, obj):
        self.handles.append(obj)
        return obj

    def parse_stream(self):
        magic   = self.read_u2()
        version = self.read_u2()
        if magic != self.STREAM_MAGIC:
            raise ValueError(f"Not a Java serialization stream (magic={hex(magic)})")
        contents = []
        while self.pos < len(self.data):
            try:
                obj = self.read_content()
                if obj is not None:
                    contents.append(obj)
            except EOFError:
                break
        return contents[0] if len(contents) == 1 else contents

    def read_content(self):
        tc = self.read_u1()
        return self.dispatch(tc)

    def dispatch(self, tc):
        if   tc == self.TC_OBJECT:       return self.read_object()
        elif tc == self.TC_STRING:        return self.read_string()
        elif tc == self.TC_LONGSTRING:    return self.read_long_string()
        elif tc == self.TC_ARRAY:         return self.read_array()
        elif tc == self.TC_CLASSDESC:     return self.read_classdesc()
        elif tc == self.TC_NULL:          return None
        elif tc == self.TC_REFERENCE:     return self.read_reference()
        elif tc == self.TC_BLOCKDATA:     return self.read_blockdata()
        elif tc == self.TC_BLOCKDATALONG: return self.read_blockdata_long()
        elif tc == self.TC_ENDBLOCKDATA:  return "__END__"
        elif tc == self.TC_ENUM:          return self.read_enum()
        elif tc == self.TC_CLASS:         return self.read_class()
        elif tc == self.TC_RESET:
            self.handles = []
            return None
        else:
            raise ValueError(f"Unknown TC byte: {hex(tc)} at pos {self.pos - 1}")

    # ── ClassDesc ──────────────────────────────────────────────────────────────

    def read_classdesc(self):
        name      = self.read_utf()
        serial_id = self.read_i8()
        cd        = {"name": name, "serial_id": serial_id, "fields": [], "super": None}
        self.new_handle(cd)
        flags = self.read_u1()
        cd["flags"] = flags
        field_count = self.read_u2()
        for _ in range(field_count):
            tc_type = chr(self.read_u1())
            fname   = self.read_utf()
            if tc_type in ("L", "["):
                # Object/array field — class name follows as string object
                class_name = self.read_content()
                cd["fields"].append((fname, tc_type, class_name))
            else:
                cd["fields"].append((fname, tc_type, None))
        self.read_content()  # classAnnotation (TC_ENDBLOCKDATA)
        super_cd = self.read_content()
        if super_cd and super_cd != "__END__":
            cd["super"] = super_cd
        return cd

    def read_class(self):
        cd = self.read_content()
        self.new_handle(cd)
        return cd

    # ── Object ─────────────────────────────────────────────────────────────────

    def read_object(self):
        cd = self.read_content()
        obj = {"__class__": cd["name"] if cd else None, "__fields__": {}}
        self.new_handle(obj)
        if cd:
            self.read_class_data(cd, obj)
        return obj

    def read_class_data(self, cd, obj):
        # Walk the class hierarchy (super first)
        if cd.get("super"):
            self.read_class_data(cd["super"], obj)
        flags = cd.get("flags", 0)
        # Read primitive + object fields declared in class
        for fname, ftype, _ in cd.get("fields", []):
            obj["__fields__"][fname] = self.read_field_value(ftype)
        # If SC_WRITE_METHOD or SC_BLOCK_DATA, read objectAnnotation
        if flags & (self.SC_WRITE_METHOD | self.SC_BLOCK_DATA):
            self.read_object_annotation(cd, obj)

    def read_field_value(self, type_code):
        if   type_code == "B": return struct.unpack(">b", self.read(1))[0]
        elif type_code == "C": return chr(self.read_u2())
        elif type_code == "D": return self.read_f8()
        elif type_code == "F": return self.read_f4()
        elif type_code == "I": return self.read_i4()
        elif type_code == "J": return self.read_i8()
        elif type_code == "S": return struct.unpack(">h", self.read(2))[0]
        elif type_code == "Z": return bool(self.read_u1())
        elif type_code in ("L", "["): return self.read_content()
        else: raise ValueError(f"Unknown field type: {type_code}")

    def read_object_annotation(self, cd, obj):
        """Read block data / objects until TC_ENDBLOCKDATA."""
        name = cd.get("name", "")
        annotation_data = []
        while True:
            tc = self.read_u1()
            if tc == self.TC_ENDBLOCKDATA:
                break
            item = self.dispatch(tc)
            if item != "__END__":
                annotation_data.append(item)

        # ── ArrayList: size stored in blockdata, elements follow ──────────────
        if "ArrayList" in name or "LinkedList" in name:
            # Size was already read as field 'size'; elements are in annotation
            elements = [x for x in annotation_data if x != "__END__"]
            obj["__elements__"] = elements

        # ── HashSet: elements are in annotation_data ──────────────────────────
        elif "HashSet" in name or "LinkedHashSet" in name:
            elements = [x for x in annotation_data if x != "__END__"]
            obj["__elements__"] = elements

        # ── HashMap / LinkedHashMap ───────────────────────────────────────────
        elif "HashMap" in name or "LinkedHashMap" in name:
            pairs = {}
            it = iter(x for x in annotation_data if x != "__END__")
            for k in it:
                try:
                    v = next(it)
                    pairs[str(k)] = v
                except StopIteration:
                    break
            obj["__map__"] = pairs

        else:
            obj["__annotation__"] = annotation_data

    # ── Primitives ─────────────────────────────────────────────────────────────

    def read_string(self):
        s = self.read_utf()
        self.new_handle(s)
        return s

    def read_long_string(self):
        s = self.read_long_utf()
        self.new_handle(s)
        return s

    def read_reference(self):
        handle = self.read_i4() - 0x7e0000
        if 0 <= handle < len(self.handles):
            return self.handles[handle]
        return f"__REF_{handle}__"

    def read_blockdata(self):
        size = self.read_u1()
        return self.read(size)

    def read_blockdata_long(self):
        size = self.read_i4()
        return self.read(size)

    def read_array(self):
        cd   = self.read_content()
        arr  = []
        self.new_handle(arr)
        size = self.read_i4()
        type_code = cd["name"][1] if cd and cd["name"].startswith("[") else "L"
        for _ in range(size):
            arr.append(self.read_field_value(type_code))
        return arr

    def read_enum(self):
        cd    = self.read_content()
        const = self.read_content()
        enum  = {"__enum__": cd["name"] if cd else None, "value": const}
        self.new_handle(enum)
        return enum


# ── JedAI object → Python dict ────────────────────────────────────────────────

def entity_profile_to_dict(obj, idx):
    """
    Convert a deserialized EntityProfile Java object to a flat dict.
    EntityProfile has:
      - entityUrl (String)
      - attributes (HashSet<Attribute>)  where Attribute has name + value
    """
    if not isinstance(obj, dict):
        return None

    fields = obj.get("__fields__", {})

    # id is the 0-based positional index — matches what the ground truth uses
    row = {"id": idx}
    row["entity_url"] = fields.get("entityUrl", "")

    # attributes field → HashSet of Attribute objects
    attr_set = fields.get("attributes", {})
    if isinstance(attr_set, dict):
        elements = attr_set.get("__elements__", [])
        for attr in elements:
            if isinstance(attr, dict):
                attr_fields = attr.get("__fields__", {})
                attr_name   = attr_fields.get("name",  "")
                attr_value  = attr_fields.get("value", "")
                if attr_name:
                    row[str(attr_name)] = str(attr_value) if attr_value else ""

    return row


def parse_profiles(filepath):
    """Deserialize *profiles binary → list of dicts."""
    with open(filepath, "rb") as f:
        data = f.read()

    d   = JavaDeserializer(data)
    obj = d.parse_stream()

    # Top-level should be an ArrayList
    elements = []
    if isinstance(obj, dict):
        elements = obj.get("__elements__", [])
    elif isinstance(obj, list):
        elements = obj

    # The Java ArrayList serialization includes the size as the first element (bytes).
    # Skip it so that real profiles start at index 0, matching the 0-based ground truth.
    if elements and isinstance(elements[0], bytes):
        elements = elements[1:]

    rows = []
    for i, ep in enumerate(elements):
        row = entity_profile_to_dict(ep, i)
        if row:
            rows.append(row)

    return rows


def parse_duplicates(filepath):
    """
    Deserialize *IdDuplicates binary → list of (id1, id2) integer pairs.
    The HashSet contains Integer objects representing duplicate entity IDs.
    In JedAI the ground truth is pairs: each element is an
    org.scify.jedai.datamodel.IdDuplicates with id1/id2 fields.
    """
    with open(filepath, "rb") as f:
        data = f.read()

    d   = JavaDeserializer(data)
    obj = d.parse_stream()

    pairs = []
    elements = []
    if isinstance(obj, dict):
        elements = obj.get("__elements__", [])
    elif isinstance(obj, list):
        elements = obj

    for item in elements:
        if isinstance(item, dict):
            fields = item.get("__fields__", {})
            # IdDuplicates has entityId1, entityId2
            id1 = fields.get("entityId1") or fields.get("id1")
            id2 = fields.get("entityId2") or fields.get("id2")
            if id1 is not None and id2 is not None:
                pairs.append((int(id1), int(id2)))

    return pairs


# ── pyjnius-based reader (preferred when JVM available) ───────────────────────

def read_with_jnius(jar_path, profiles_path, duplicates_path):
    """Use pyjnius + JedAI jar to read files natively. Returns (rows, pairs)."""
    import jnius_config
    jnius_config.add_classpath(str(jar_path))
    from jnius import autoclass

    # ── Profiles ──────────────────────────────────────────────────────────────
    Reader = autoclass("org.scify.jedai.datareader.entityreader.EntitySerializationReader")
    reader = Reader(str(profiles_path))
    profiles = reader.getEntityProfiles()

    rows = []
    it = profiles.iterator()
    idx = 0
    while it.hasNext():
        ep = it.next()
        row = {"id": idx, "entity_url": ep.getEntityUrl()}
        attrs_it = ep.getAttributes().iterator()
        while attrs_it.hasNext():
            attr = attrs_it.next()
            row[attr.getName()] = attr.getValue()
        rows.append(row)
        idx += 1

    # ── Ground truth ──────────────────────────────────────────────────────────
    GTReader = autoclass("org.scify.jedai.datareader.groundtruthreader.GtSerializationReader")
    gt_reader = GTReader(str(duplicates_path))
    gt_reader.setEntityIds(0, profiles.size())
    duplicates = gt_reader.getDuplicatePairs(None)

    pairs = []
    gt_it = duplicates.iterator()
    while gt_it.hasNext():
        pair = gt_it.next()
        pairs.append((pair.getEntityId1(), pair.getEntityId2()))

    return rows, pairs


# ── Save to disk ──────────────────────────────────────────────────────────────

def save_outputs(rows, pairs, out_dir, size_label, fmt):
    import pandas as pd

    out_dir = Path(out_dir) / size_label
    out_dir.mkdir(parents=True, exist_ok=True)

    df_profiles = pd.DataFrame(rows)
    df_gt       = pd.DataFrame(pairs, columns=["id1", "id2"])

    print(f"  Profiles   : {len(df_profiles):,} rows × {len(df_profiles.columns)} cols")
    print(f"  Columns    : {list(df_profiles.columns)}")
    print(f"  Duplicates : {len(df_gt):,} pairs")

    if fmt in ("csv", "both"):
        p_path = out_dir / "profiles.csv"
        g_path = out_dir / "ground_truth.csv"
        df_profiles.to_csv(p_path, index=False)
        df_gt.to_csv(g_path, index=False)
        print(f"  Saved CSV  → {p_path}")
        print(f"             → {g_path}")

    if fmt in ("parquet", "both"):
        p_path = out_dir / "profiles.parquet"
        g_path = out_dir / "ground_truth.parquet"
        df_profiles.to_parquet(p_path, index=False)
        df_gt.to_parquet(g_path, index=False)
        print(f"  Saved PQ   → {p_path}")
        print(f"             → {g_path}")

    return df_profiles, df_gt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas is required:  pip install pandas pyarrow")
        sys.exit(1)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    jar_path   = Path(args.jar)

    if not data_dir.exists():
        print(f"[ERROR] data_dir not found: {data_dir}")
        sys.exit(1)

    # Decide method
    use_jnius = False
    if not args.fallback:
        if not jar_path.exists():
            print(f"[WARN] JAR not found at: {jar_path}")
            print(f"       Download it from:")
            print(f"       https://github.com/scify/JedAIToolkit/releases")
            print(f"       Falling back to pure-Python parser.\n")
        else:
            try:
                import jnius_config
                use_jnius = True
                print("[INFO] Using pyjnius + JedAI JAR for reading.\n")
            except ImportError:
                print("[WARN] pyjnius not installed. Falling back to pure-Python parser.")
                print("       To install: pip install pyjnius\n")

    if not use_jnius:
        print("[INFO] Using pure-Python Java deserializer.\n")

    summary = []

    for size in args.sizes:
        profiles_path   = data_dir / f"{size}profiles"
        duplicates_path = data_dir / f"{size}IdDuplicates"

        if not profiles_path.exists():
            print(f"[SKIP] {size}profiles not found in {data_dir}")
            continue
        if not duplicates_path.exists():
            print(f"[SKIP] {size}IdDuplicates not found in {data_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  Converting: {size}")
        print(f"{'='*60}")

        try:
            if use_jnius:
                rows, pairs = read_with_jnius(jar_path, profiles_path, duplicates_path)
            else:
                print("  Parsing profiles...")
                rows = parse_profiles(profiles_path)
                print("  Parsing ground truth...")
                pairs = parse_duplicates(duplicates_path)

            df_p, df_gt = save_outputs(rows, pairs, output_dir, size, args.format)
            summary.append({
                "size":       size,
                "profiles":   len(df_p),
                "columns":    len(df_p.columns),
                "duplicates": len(df_gt),
                "status":     "OK"
            })

        except Exception as e:
            print(f"  [ERROR] {size}: {e}")
            traceback.print_exc()
            summary.append({"size": size, "status": f"ERROR: {e}"})

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  CONVERSION SUMMARY")
    print(f"{'='*60}")
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    print(f"\nOutput directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()