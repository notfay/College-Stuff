package model;

public class Motor extends Kendaraan {

	public Motor(String id, String merek, int tahun) {
		super(id, merek, tahun);
		// TODO Auto-generated constructor stub
	}

	@Override
	public String getTipe() {
		// TODO Auto-generated method stub
		return "Motor";
	}

}
