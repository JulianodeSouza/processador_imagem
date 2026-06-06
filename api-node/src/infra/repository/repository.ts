type Parameters = {
  model: any;
  includes?: any[];
  params?: object;
  type?: string;
  replacements?: object;
  data?: object;
  query?: string;
};

export default class RepositoryModel {
  protected async getAll<T>(parameters: Parameters): Promise<T[]> {
    const itemsList = await parameters.model.findAll({
      where: parameters.params || {},
      include: parameters.includes,
      order: [["name", "ASC"]],
    });

    return itemsList;
  }

  protected async getOne<T>(parameters: Parameters): Promise<T | null> {
    const itemList = await parameters.model.findOne({
      where: parameters.params || {},
      include: parameters.includes,
    });

    return itemList;
  }

  protected async save(parameters: Parameters) {
    await parameters.model.save();
  }

  protected async update(parameters: Parameters): Promise<void> {
    await parameters.model.update(parameters.data, {
      where: parameters.params,
    });
  }

  protected async delete(parameters: Parameters): Promise<void> {
    await parameters.model.update(parameters.data, {
      where: parameters.params,
    });
  }

  protected async executeQuery<T>(parameters: Parameters): Promise<T[]> {
    const result = await parameters.model.sequelize.query(parameters.query, {
      type: parameters.type,
      replacements: parameters.replacements,
    });

    return result;
  }
}
