package model;

public abstract class Kendaraan {
	
	private String merk;

	public Kendaraan(String merk) {
		super();
		this.merk = merk;
	}

	public String getMerk() {
		return merk;
	}

	public void setMerk(String merk) {
		this.merk = merk;
	}
	
	public abstract void info();
	
}
